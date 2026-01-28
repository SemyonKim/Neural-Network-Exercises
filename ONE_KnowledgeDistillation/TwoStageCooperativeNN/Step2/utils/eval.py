"""
eval.py

Evaluation utilities for cooperative neural network experiments.

Functions:
- accuracy: computes precision@k (top-k accuracy) for model predictions
"""

__all__ = ["accuracy"]

import torch


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> list:
    """
    Compute the precision@k (top-k accuracy) for the specified values of k.

    Args:
        output (torch.Tensor): Model outputs (logits or probabilities), shape [batch_size, num_classes].
        target (torch.Tensor): Ground truth labels, shape [batch_size].
        topk (tuple): Tuple of k values (e.g., (1, 5)) for which to compute accuracy.

    Returns:
        list: List of accuracies corresponding to each k in topk.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    # Get top-k predictions
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # transpose for comparison

    # Compare predictions with targets
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
