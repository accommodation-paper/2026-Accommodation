import torch
from torch import nn


def likelihood_policy(likelihood: torch.Tensor, accommodation_layer: nn.Module, eps: float = 1e-12, is_neg=False):
	return likelihood
