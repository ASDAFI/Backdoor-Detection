# Backdoor Detection

This research project focuses on **detecting backdoor attacks** in the **PreActResNet architecture** under **data-limited configurations** within a **1-minute time limit** (tested on an **NVIDIA RTX 4090**).

## Problem Statement

The complete problem description is available in the [`problem_statement.pdf`](problem_statement.pdf).

## Explorations

Various ideas and methodologies explored during this research are documented in the `analysis` directory. **Before running the notebooks**, ensure they are moved to the project's root directory.

## Usage

The main detection method is implemented in `main.py` with the following function:

```python
def backdoor_model_detector(model: PreActResNet, 
                            num_classes: int,
                            test_images_folder_address: str,
                            transformation: transforms.Compose
                           ) -> bool:
    """
    Detects whether the given model is malicious (contains a backdoor) or not.
    """
    # Function implementation
