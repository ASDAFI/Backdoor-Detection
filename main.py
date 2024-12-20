import glob
import os
from collections import defaultdict
import logging
from typing import List, Dict, Tuple
import random
import warnings
import time
import functools
import sys
from statistics import mean
import gc

import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
from scipy import stats

import torch
from torch.optim import Adam

LOGGER = logging.getLogger('detector')
LOGGER.setLevel(logging.INFO)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

LOGGER.addHandler(stream_handler)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f'device is {DEVICE}')


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        LOGGER.info(f"{func.__name__} took {end - start:.6f} seconds")
        return result

    return wrapper


def transform_images(images_path: List[str], transformation: transforms.Compose):
    transformed_images = []
    for img_path in images_path:
        try:
            image = Image.open(img_path).convert('RGB')
            image = transformation(image)
            transformed_images.append(image)
        except Exception as e:
            LOGGER.error(f"Error loading image {img_path}: {e}")

    if not transformed_images:
        LOGGER.error("No images were loaded. Please check the images_path list.")

    return torch.stack(transformed_images)


def extract_normalization_params(transformation: transforms.Compose):
    mean = None
    std = None

    for transform in transformation.transforms:
        if isinstance(transform, transforms.Normalize):
            mean = transform.mean
            std = transform.std
    return mean, std


def get_logits_and_probs(model: PreActResNet, transformed_images: torch.Tensor):
    transformed_images = transformed_images.to(DEVICE)
    logits = model(transformed_images)
    probabilities = F.softmax(logits, dim=1)
    return probabilities, logits


def calculate_margins(probs: List[torch.Tensor],
                      labels: List[int]) -> Tuple[defaultdict[int, List[float]],
defaultdict[int, List[float]]]:
    accepted_margins = defaultdict(list)
    failed_margins = defaultdict(list)
    for i in range(len(probs)):
        topk = torch.topk(probs[i], k=2, largest=True, sorted=True)
        topk_values = topk.values

        margin = topk_values[0].item() - topk_values[1].item()
        if labels[i] == torch.argmax(probs[i]).item():
            accepted_margins[labels[i]].append(margin)
        else:
            failed_margins[torch.argmax(probs[i]).item()].append(margin)
    return accepted_margins, failed_margins


def find_safe_margin(accepted_margins: defaultdict[int, List[float]],
                     failed_margins: defaultdict[int, List[float]]):
    min_accepted_margins = dict()
    for c in accepted_margins.keys():
        max_failed = float('inf')
        if failed_margins[c]:
            max_failed = max(failed_margins[c])

        min_accepted_margin = max_failed
        for margin in accepted_margins[c]:
            if margin <= min_accepted_margin:
                min_accepted_margin = margin

        min_accepted_margins[c] = min_accepted_margin
    return min_accepted_margins


def project_image(optimized_img: torch.Tensor, mean: float, std: float):
    mean = torch.tensor(mean).view(-1, 1, 1).to(optimized_img.device)
    std = torch.tensor(std).view(-1, 1, 1).to(optimized_img.device)

    min_val = (0.0 - mean) / std
    max_val = (1.0 - mean) / std

    optimized_img = torch.clamp(optimized_img, min=min_val, max=max_val)

    return optimized_img


def project_image_with_epsilon(
        optimized_img: torch.Tensor,
        reference_img: torch.Tensor,
        mean: float,
        std: float,
        epsilon: float
) -> torch.Tensor:
    mean_tensor = torch.tensor(mean).view(-1, 1, 1).to(optimized_img.device)
    std_tensor = torch.tensor(std).view(-1, 1, 1).to(optimized_img.device)
    normalized_ref = (reference_img - mean_tensor) / std_tensor

    min_val = (0.0 - mean_tensor) / std_tensor
    max_val = (1.0 - mean_tensor) / std_tensor
    optimized_img = torch.clamp(optimized_img, min=min_val, max=max_val)

    # optimized_img = torch.max(optimized_img, normalized_ref - epsilon)
    # optimized_img = torch.min(optimized_img, normalized_ref + epsilon)

    return optimized_img


def select_top_images_per_class(probs: List[torch.Tensor],
                                images: List[torch.Tensor],
                                labels: List[int],
                                num_classes: int, top_k=3) -> defaultdict[int, List[torch.Tensor]]:
    selected_images_per_class = defaultdict(list)

    with torch.no_grad():
        for idx, (prob, label) in enumerate(zip(probs, labels)):
            detected_class = torch.argmax(prob).item()
            if detected_class == label:
                confidence = prob[detected_class].item()
                selected_images_per_class[label].append((confidence, images[idx]))

    for c in range(num_classes):
        class_images = selected_images_per_class[c]
        sorted_images = sorted(class_images, key=lambda x: x[0], reverse=True)
        selected_images_per_class[c] = [img for _, img in sorted_images[:top_k]]
    return selected_images_per_class


def generate_random_image(transformation):
    transformation = transforms.Compose([transforms.ToPILImage()] + list(transformation.transforms))
    return transformation(torch.randn(3, *(220, 220)) * 255)


def compute_max_margin_parallel_with_epsilon(
        model,
        selected_images_per_class,
        reference_images_per_class,
        num_classes,
        projection_mean,
        projection_std,
        epsilon,
        max_iterations=1000,
        lr=0.01,
        tolerance=1e-5,
        max_img_per_class=3,
        device=DEVICE,
        parallel_classes=5
):
    MAX_SECONDS = 55
    ZERO = 0.0001
    start_time = time.time()
    model.eval()
    max_margins = {}
    all_margins_per_class = defaultdict(list)
    all_triggers_per_class = defaultdict(list)
    class_indices = list(range(num_classes))
    class_batches = [
        class_indices[i:i + parallel_classes]
        for i in range(0, len(class_indices), parallel_classes)
    ]
    BREAKED = False
    for batch_num, class_batch in enumerate(class_batches):
        elapsed_time = time.time() - start_time
        if BREAKED or elapsed_time >= MAX_SECONDS:
            BREAKED = True
            for c in class_batch:
                all_triggers_per_class[c] = []
                all_margins_per_class[c] = [ZERO * random.random() for _ in range(max_img_per_class)]
            continue

        batch_images = []
        batch_reference_images = []
        batch_class_labels = []
        class_to_image_indices = defaultdict(list)
        for c in class_batch:
            images_to_optimize = []
            reference_images = []
            for k in range(num_classes):
                if k == c:
                    continue
                images = selected_images_per_class.get(k, [])
                references = reference_images_per_class.get(k, [])
                if not images:
                    continue
                if len(references) < len(images):
                    raise ValueError(f"Not enough reference images for class {k}.")
                images_to_optimize.extend(images)
                reference_images.extend(references[:len(images)])
            random.shuffle(images_to_optimize)
            images_to_optimize = images_to_optimize[:max_img_per_class]
            reference_images = reference_images[:max_img_per_class]
            if not images_to_optimize:
                max_margins[c] = 0.0
                continue
            start_idx = len(batch_images)
            batch_images.extend([img.clone().detach() for img in images_to_optimize])
            batch_reference_images.extend([ref.clone().detach() for ref in reference_images])
            batch_class_labels.extend([c] * len(images_to_optimize))
            for i in range(len(images_to_optimize)):
                class_to_image_indices[c].append(start_idx + i)
        if not batch_images:
            continue
        optimized_imgs = torch.stack(batch_images).to(device)
        reference_imgs = torch.stack(batch_reference_images).to(device)
        optimized_imgs.requires_grad = True
        optimizer = Adam([optimized_imgs], lr=lr) if lr is not None else Adam([optimized_imgs])
        batch_size = optimized_imgs.size(0)
        max_margins_batch = torch.full((batch_size,), -float('inf'), device=device)
        triggers_batch = optimized_imgs.clone().detach()
        f_old = torch.full((batch_size,), float('inf'), device=device)
        for iteration in range(max_iterations):
            elapsed_time = time.time() - start_time
            if BREAKED or elapsed_time >= MAX_SECONDS:
                BREAKED = True
                break
            optimizer.zero_grad()
            probs, logits = get_logits_and_probs(model, optimized_imgs)
            target_classes = torch.tensor(batch_class_labels, device=device)
            g_c = logits[torch.arange(batch_size), target_classes]
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[torch.arange(batch_size), target_classes] = False
            g_k, _ = torch.max(logits.masked_fill(~mask, -float('inf')), dim=1)
            margin = g_c - g_k
            update_mask = margin > max_margins_batch
            max_margins_batch = torch.maximum(max_margins_batch, margin)
            triggers_batch[update_mask] = optimized_imgs.detach()[update_mask]
            loss = -margin.mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                optimized_imgs.copy_(project_image_with_epsilon(
                    optimized_imgs,
                    reference_imgs,
                    projection_mean,
                    projection_std,
                    epsilon
                ))
            relative_change = torch.abs(margin - f_old) / (torch.abs(f_old) + 1e-8)
            if torch.all(relative_change < tolerance):
                break
            f_old = margin.clone()
            if (iteration + 1) % 100 == 0:
                pass
        for c in class_batch:
            image_indices = class_to_image_indices.get(c, [])
            if not image_indices:
                continue
            class_margins = max_margins_batch[image_indices].cpu().tolist()
            class_triggers = triggers_batch[image_indices].detach().cpu()
            all_margins_per_class[c].extend([max(margin_, ZERO * random.random()) for margin_ in class_margins])
            all_triggers_per_class[c].extend(class_triggers)
            max_margins[c] = max(max(class_margins), ZERO * random.random())
        del optimized_imgs, reference_imgs
        if not BREAKED:
            del logits, probs, margin, loss, g_c, g_k
        del triggers_batch, max_margins_batch, f_old, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    if max_margins:
        avg_margin = mean([max_margins.get(c) for c in max_margins.keys()])
    else:
        avg_margin = ZERO * random.random()
    max_margins_sorted = [max_margins.get(c, avg_margin) for c in range(num_classes)]
    return max_margins_sorted, all_triggers_per_class, all_margins_per_class


def compute_p_values(gamma_list: List[float],
                     distributions=['gamma', 'norm'], p_value_type='standard') -> dict[str, float]:
    if not gamma_list:
        raise ValueError("gamma_list is empty.")

    gamma_array = np.array(gamma_list)
    r_max = np.max(gamma_array)
    n = len(gamma_array)

    null_data = gamma_array[gamma_array != r_max]
    if len(null_data) == 0:
        raise ValueError("All values in gamma_list are identical.")

    p_values = {}

    for dist_name in distributions:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if dist_name == 'gamma':
                    a, loc, scale = stats.gamma.fit(null_data, floc=0)
                    fitted_dist = stats.gamma(a, loc=loc, scale=scale)
                elif dist_name == 'norm':
                    mu, sigma = stats.norm.fit(null_data)
                    fitted_dist = stats.norm(loc=mu, scale=sigma)
                elif dist_name == 'expon':
                    loc, scale = stats.expon.fit(null_data)
                    fitted_dist = stats.expon(loc=loc, scale=scale)
                elif dist_name == 'beta':
                    a, b, loc, scale = stats.beta.fit(null_data, floc=0, fscale=1)
                    fitted_dist = stats.beta(a, b, loc=loc, scale=scale)
                elif dist_name == 'lognorm':
                    s, loc, scale = stats.lognorm.fit(null_data, floc=0)
                    fitted_dist = stats.lognorm(s, loc=loc, scale=scale)
                else:
                    LOGGER.error(f"Distribution '{dist_name}' is not supported.")
                    continue

                H0_r_max = fitted_dist.cdf(r_max)

                if p_value_type == 'standard':
                    p_val = 1 - H0_r_max ** n
                elif p_value_type == 'user_specified':
                    p_val = H0_r_max ** (n - 1)
                else:
                    LOGGER.error(
                        f"p_value_type '{p_value_type}' is not recognized. Choose 'standard' or 'user_specified'.")
                    continue

                p_values[dist_name] = p_val

        except Exception as e:
            LOGGER.error(f"An error occurred while fitting distribution '{dist_name}': {e}")
            p_values[dist_name] = None

    return p_values


@timeit
def backdoor_model_detector(model: PreActResNet, num_classes: int,
                            test_images_folder_address: str,
                            transformation: transforms.Compose
                            ) -> bool:
    t = time.time()
    model = model.to(DEVICE)
    model.eval()

    images_path = glob.glob(os.path.join(test_images_folder_address, '*.jpg'))
    labels = [int(image_path.split('_')[-1].split('.')[0]) for image_path in images_path]

    projection_mean, projection_std = extract_normalization_params(transformation)
    transformed_images = transform_images(images_path, transformation)

    probs, logits = get_logits_and_probs(model, transformed_images)
    accepted_margins, failed_margins = calculate_margins(logits, labels)
    safe_margins = [x[1] for x in
                    sorted(find_safe_margin(accepted_margins, failed_margins).items(),
                           key=lambda x: x[0])]

    k = max(int(9 * (10 / num_classes)), 1)
    max_iterations = int(1500 * (10 / num_classes))
    max_img_per_class = max(int(9 * (10 / num_classes)), 1)

    epsilon = 3
    confident_images_per_class = select_top_images_per_class(probs, transformed_images,
                                                             labels, num_classes, top_k=k)
    lr = 0.1
    tolerance = 1e-5

    if max_img_per_class <= 2:
        confident_images_per_class = {c: [generate_random_image(transformation) for i in range(k)]
                                      for c in range(num_classes)}
        epsilon = 100
        lr = 0.1
        tolerance = 1e-3

    max_margins, triggers_per_class, all_margins_per_class = compute_max_margin_parallel_with_epsilon(model,
                                                                                                      confident_images_per_class,
                                                                                                      confident_images_per_class,
                                                                                                      num_classes,
                                                                                                      projection_mean,
                                                                                                      projection_std,
                                                                                                      max_iterations=max_iterations,
                                                                                                      lr=lr,
                                                                                                      tolerance=tolerance,
                                                                                                      epsilon=epsilon,
                                                                                                      max_img_per_class=max_img_per_class)

    for i in range(len(max_margins)):
        max_margins[i] = max(max_margins[i], 0)
    average_margins = list()

    for c in range(num_classes):
        for i in range(len(all_margins_per_class[c])):
            all_margins_per_class[c][i] = max(all_margins_per_class[c][i], 0)
        average_margins.append(mean(all_margins_per_class[c]))

    LOGGER.info(f'finding adversaries has taken {round(time.time() - t, 2)} seconds')
    target_class_c = max_margins.index(max(max_margins))
    LOGGER.info(f'working on class {target_class_c}')
    p_values_standard = compute_p_values(max_margins, distributions=['gamma', 'norm', 'expon'],
                                         p_value_type='standard')
    avg_p_values_standard = compute_p_values(average_margins, distributions=['gamma', 'norm', 'expon'],
                                             p_value_type='standard')
    safe_p_values_standard = compute_p_values(safe_margins, distributions=['gamma', 'norm', 'expon'],
                                              p_value_type='standard')
    LOGGER.info(f'max margin p values {p_values_standard}')
    LOGGER.info(f'average maximum margins p_values {avg_p_values_standard}')
    LOGGER.info(f'safe margins p values {safe_p_values_standard}')

    THRESHOLD = 0.06
    if avg_p_values_standard['gamma'] <= THRESHOLD:
        LOGGER.info(f'detected as a malicious model')
    else:
        LOGGER.info(f'detected as a clean model')
    return avg_p_values_standard['gamma'] <= THRESHOLD