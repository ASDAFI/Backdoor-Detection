# Backdoor Detection

This research project focuses on **detecting backdoor attacks** in the **PreActResNet architecture** under **data-limited configurations** within a **1-minute time limit** (tested on an **NVIDIA RTX 4090**).

## Problem Statement

The complete problem description is available in the [`problem_statement.pdf`](problem_statement.pdf).

## Explorations

Various ideas and methodologies explored during this research are documented in the `analysis` directory. **Before running the notebooks**, ensure they are moved to the project's root directory.

## Usage

The main detection method is implemented in `main.py` with the following function:

```
def backdoor_model_detector(model: PreActResNet, 
                            num_classes: int,
                            test_images_folder_address: str,
                            transformation: transforms.Compose
                           ) -> bool:
    """
    Detects whether the given model is malicious (contains a backdoor) or not.
    """
    # Function implementation
```
    
This function analyzes the provided `PreActResNet` model to determine if it has been compromised by a backdoor attack.

## Related Works

Interesting articles and novel ideas are stored in the `related_works` directory.

## Results

This model was developed for the [Rayan Trustworthy AI Challenge](https://ai.rayan.global), achieving a score of **65/100** and ranking **12th**.
