import numpy as np
import torch

from accommodation.experiments.helpers.latent_metrics import calculate_nc1_details
from accommodation.model.compatibility_operator import compatibility_operator


def _extract_features(model, x):
	if hasattr(model, "encode"):
		attention_mask = (x != 0).long()
		return model.encode(x, attention_mask)
	if hasattr(model, "backbone"):
		return model.backbone(x)
	if hasattr(model, "model"):
		return model.model(x)
	raise AttributeError("Unsupported model: cannot extract latent features")


def _effective_compatibility_weights_from_features(layer, features, eps=1e-12):
	mu_sources = [layer.positive_mu, layer.negative_mu]
	sigma_sources = [layer.positive_sigma, layer.negative_sigma]
	mu_all = torch.cat([tensor for tensor in mu_sources if tensor is not None], dim=0)
	sigma_all = torch.cat([tensor for tensor in sigma_sources if tensor is not None], dim=0)
	x_mu = layer.mu_encoder(features).unsqueeze(1).unsqueeze(2)
	x_sigma = layer.sigma_encoder(features).unsqueeze(1).unsqueeze(2)
	compatibility = compatibility_operator(
		x_mu,
		x_sigma,
		mu_all.unsqueeze(0),
		sigma_all.unsqueeze(0),
	).clamp(0.0, 1.0)
	compatibility = compatibility.mean(dim=-1).reshape(features.size(0), -1)
	return compatibility / (compatibility.sum(dim=1, keepdim=True) + eps)


@torch.no_grad()
def compute_epoch_visualization_stats(model, dataloader, device, eps=1e-12):
	model.eval()
	layer = model.accommodation_layer
	embeddings = []
	labels = []
	total_usage_entropy = 0.0
	total_samples = 0
	sum_kappa_total = 0.0
	sum_kappa_type = {"Positive": 0.0, "Negative": 0.0}
	sum_frac = {"Positive": 0.0, "Negative": 0.0}
	seen_types = set()

	for x, y in dataloader:
		x = x.to(device)
		y = y.to(device)
		features = _extract_features(model, x)
		embeddings.append(features.detach().cpu().numpy())
		labels.append(y.detach().cpu().numpy())

		weights = _effective_compatibility_weights_from_features(layer, features, eps=eps)
		entropy = -(weights * (weights + eps).log()).sum(dim=1)
		usage_entropy = torch.exp(entropy)
		total_usage_entropy += usage_entropy.sum().item()

		sizes = {}
		if layer.positive_mu is not None:
			classes, potents_per_class, _ = layer.positive_mu.shape
			sizes["Positive"] = classes * potents_per_class
		if layer.negative_mu is not None:
			classes, potents_per_class, _ = layer.negative_mu.shape
			sizes["Negative"] = classes * potents_per_class

		seen_types.update(sizes)
		weight_square_sum = weights.pow(2).sum(dim=1) + eps
		kappa_total = 1.0 / weight_square_sum
		sum_kappa_total += kappa_total.sum().item()

		start = 0
		for potent_type in ("Positive", "Negative"):
			if potent_type not in sizes:
				continue
			end = start + sizes[potent_type]
			type_weights = weights[:, start:end]
			type_weight_square_sum = type_weights.pow(2).sum(dim=1)
			frac = type_weight_square_sum / weight_square_sum
			sum_frac[potent_type] += frac.sum().item()
			sum_kappa_type[potent_type] += (frac * kappa_total).sum().item()
			start = end
		total_samples += x.size(0)

	embeddings = np.concatenate(embeddings, axis=0)
	labels = np.concatenate(labels, axis=0)
	nc1_details = calculate_nc1_details(embeddings, labels)

	return {
		"effective_support": {
			"kappa_total": sum_kappa_total / total_samples,
			"kappa_part": {name: sum_kappa_type[name] / total_samples for name in seen_types},
			"kappa_frac": {name: sum_frac[name] / total_samples for name in seen_types},
		},
		"usage_entropy": total_usage_entropy / total_samples,
		"nc1": nc1_details["nc1"],
		"nc1_components": nc1_details,
		"differentiation_matrix": layer.calc_differentiation_tensor().mean(dim=-1).detach().cpu().numpy().tolist(),
	}
