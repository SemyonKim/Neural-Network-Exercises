"""
branchreplacement.py

Utilities for branch replacement and network surgery in cooperative neural networks.

Functions:
- get_layers_name_of_branch: return parameter names for a given branch ID
- get_control_v1_box: merge parameters from multiple branch models into a control dictionary
- compare_exact: check if two parameter dictionaries are exactly equal
- compare_approximate: check if two parameter dictionaries are approximately equal
"""

import copy
import numpy as np


def get_layers_name_of_branch(id_branch: str = "1") -> list:
    """
    Return the list of parameter names for a given branch ID.

    Args:
        id_branch (str): Branch identifier (e.g., '1', '2', '3').

    Returns:
        list: List of parameter names for the branch.
    """
    if not isinstance(id_branch, str):
        raise TypeError("Error: id_branch must be a string!")

    layers = []

    # First block (layer3.<id>.0)
    base_block = [
        "conv1.weight", "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var", "bn1.num_batches_tracked",
        "conv2.weight", "bn2.weight", "bn2.bias", "bn2.running_mean", "bn2.running_var", "bn2.num_batches_tracked",
        "downsample.0.weight", "downsample.1.weight", "downsample.1.bias",
        "downsample.1.running_mean", "downsample.1.running_var", "downsample.1.num_batches_tracked"
    ]
    layers.extend([f"module.layer3_{id_branch}.0.{name}" for name in base_block])

    # Remaining blocks (layer3.<id>.1–4)
    for i in range(1, 5):
        block = [
            "conv1.weight", "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var", "bn1.num_batches_tracked",
            "conv2.weight", "bn2.weight", "bn2.bias", "bn2.running_mean", "bn2.running_var", "bn2.num_batches_tracked"
        ]
        layers.extend([f"module.layer3_{id_branch}.{i}.{name}" for name in block])

    # Classifier
    layers.extend([
        f"module.classfier3_{id_branch}.weight",
        f"module.classfier3_{id_branch}.bias"
    ])

    return layers


def get_control_v1_box(control_v1_model1: dict,
                       control_v1_model2: dict,
                       control_v1_model3: dict,
                       model1_branch_id: int,
                       model2_branch_id: int,
                       model3_branch_id: int,
                       control_v1_list: list) -> dict:
    """
    Merge parameters from three branch models into a control dictionary.

    Args:
        control_v1_model1, control_v1_model2, control_v1_model3 (dict): Parameter dictionaries.
        model1_branch_id, model2_branch_id, model3_branch_id (int): Branch IDs for each model.
        control_v1_list (list): List of parameter keys to merge.

    Returns:
        dict: Merged control dictionary.
    """
    if not all(isinstance(x, int) for x in [model1_branch_id, model2_branch_id, model3_branch_id]):
        raise TypeError("Error: branch IDs must be integers!")

    control_v1_box = copy.deepcopy(control_v1_model1)

    for key in control_v1_list:
        if key == "module.bn_v1.num_batches_tracked":
            control_v1_box[key] = (control_v1_model1[key] +
                                   control_v1_model2[key] +
                                   control_v1_model3[key]) / 3
        else:
            control_v1_box[key][0] = control_v1_model1[key][model1_branch_id - 1]
            control_v1_box[key][1] = control_v1_model2[key][model2_branch_id - 1]
            control_v1_box[key][2] = control_v1_model3[key][model3_branch_id - 1]

    return control_v1_box


def compare_exact(first: dict, second: dict) -> bool:
    """
    Check if two dictionaries of numpy arrays are exactly equal.

    Args:
        first, second (dict): Dictionaries to compare.

    Returns:
        bool: True if exactly equal, False otherwise.
    """
    if first.keys() != second.keys():
        return False
    return all(np.array_equal(first[k], second[k]) for k in first)


def compare_approximate(first: dict, second: dict) -> bool:
    """
    Check if two dictionaries of numpy arrays are approximately equal.

    Args:
        first, second (dict): Dictionaries to compare.

    Returns:
        bool: True if approximately equal, False otherwise.
    """
    if first.keys() != second.keys():
        return False
    return all(np.allclose(first[k], second[k]) for k in first)
