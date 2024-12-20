import os
import glob

import torch

import pre_act_resnet as PreActResNet
from main import backdoor_model_detector

ROOT_DIR = 'eval_dataset'

DEVICE = 'cuda'



def load_model(num_classes, model_path):
    model = PreActResNet.PreActResNet18(num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)
    model.eval()

    return model


def load_test(idx: int):
    test_root_dir = os.path.join(ROOT_DIR, str(idx))

    metadata = torch.load(os.path.join(test_root_dir, 'metadata.pt'))

    num_classes = metadata['num_classes']
    ground_truth = metadata['ground_truth']
    images_root_dir = metadata['test_images_folder_address']
    transformation = metadata['transformation']

    model_path = os.path.join(test_root_dir, 'model.pt')

    if images_root_dir[0] == '.':
        images_root_dir = images_root_dir[2:]


    model = load_model(num_classes, model_path)

    return model, num_classes, ground_truth, transformation, images_root_dir

y = []
pred = []

for i in range(50):
    model, num_classes, ground_truth, transformation, images_root_dir = load_test(i)
    y.append(ground_truth)

    pred.append(backdoor_model_detector(model, num_classes, images_root_dir, transformation))

