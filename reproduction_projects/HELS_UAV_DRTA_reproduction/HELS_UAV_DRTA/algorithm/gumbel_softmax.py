"""
Gumbel-Softmax: 离散动作空间的可微采样 (Eq.41)
"""
import torch
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-10, device='cpu'):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(logits, temperature=1.0, hard=False):
    """
    Gumbel-Softmax with optional Straight-Through (ST) for discrete actions.

    Args:
        logits: (..., n_actions)
        temperature: >0, smaller → closer to argmax
        hard: if True, forward=one_hot, backward=softmax gradient
    Returns:
        (..., n_actions) soft or hard one-hot sample
    """
    g = sample_gumbel(logits.shape, device=logits.device)
    y_soft = F.softmax((logits + g) / temperature, dim=-1)

    if hard:
        idx = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        return (y_hard - y_soft).detach() + y_soft  # ST gradient
    return y_soft
