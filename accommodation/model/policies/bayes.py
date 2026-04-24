import torch
from torch import nn


def _class_prior_by_variance(dirac_layer: nn.Module, device: str | torch.device | None = None, prior_floor: float = 1e-12, is_neg=False):
	if device is None: device = dirac_layer.positive_sigma.device
	dev = torch.device(device)
	if is_neg:
		sigmas = dirac_layer.negative_sigma.to(dev)
	else:
		sigmas = dirac_layer.positive_sigma.to(dev)
	areas = sigmas
	areas = torch.clamp(areas, min=0.0)

	denom = areas.sum(dim=0, keepdim=True).clamp(min=prior_floor)
	prior = (areas / denom).clamp(min=prior_floor)
	return prior


def bayes_policy(likelihood: torch.Tensor, dirac_layer: nn.Module, is_neg=False):
	likelihood = likelihood
	prior = _class_prior_by_variance(dirac_layer, is_neg=is_neg).to(likelihood.device, dtype=likelihood.dtype)
	return likelihood * prior.unsqueeze(0)
