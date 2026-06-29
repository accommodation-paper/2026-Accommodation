import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from accommodation.datasets.vision.cifar10 import Cifar10Dataset
from accommodation.datasets.vision.fashion_mnist import FashionMNISTDataset
from accommodation.datasets.vision.mnist import MNISTDataset
from accommodation.datasets.text.agnews import AGNewsDataset
from accommodation.datasets.text.imdb import IMDBDataset
from accommodation.datasets.text.yelp_reviews import YelpReviewsStarsDataset
from accommodation.experiments.helpers.visualization_cache import compute_epoch_visualization_stats
from accommodation.experiments.classifiers.cifar_accommodation_classifier import instantiate_cifar_accommodation_classifier
from accommodation.experiments.classifiers.gru_accommodation_classifier import instantiate_gru_accommodation_classifier
from accommodation.experiments.classifiers.mnist_accommodation_classifier import instantiate_mnist_accommodation_classifier
from accommodation.model.compatibility_operator import compatibility_operator


IMAGE_DATASETS = {"MNIST", "FMNIST", "CIFAR10"}


def _resolve_device(device=None):
	if device is not None:
		return device
	if torch.cuda.is_available():
		return "cuda"
	if torch.backends.mps.is_available():
		return "mps"
	return "cpu"


def _json_ready(value):
	if isinstance(value, torch.Tensor):
		return _json_ready(value.detach().cpu().numpy())
	if isinstance(value, np.ndarray):
		return _json_ready(value.tolist())
	if isinstance(value, np.generic):
		return _json_ready(value.item())
	if isinstance(value, float):
		return None if not np.isfinite(value) else value
	if isinstance(value, dict):
		return {key: _json_ready(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_json_ready(item) for item in value]
	return value


def _trace_json_paths(results_dir: Path, experiment_type: str, dataset: str | None = None):
	paths = []
	for config_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
		type_dir = config_dir / experiment_type
		if not type_dir.exists():
			continue
		for dataset_dir in sorted(path for path in type_dir.iterdir() if path.is_dir()):
			if dataset is not None and dataset_dir.name != dataset:
				continue
			paths.extend(sorted(path for path in dataset_dir.glob("*.json") if path.is_file()))
	return paths


def _load_json(path: Path):
	with open(path, "r", encoding="utf-8") as file:
		return json.load(file)


def _write_json(path: Path, data):
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as file:
		json.dump(data, file, indent=2)


def _calculate_drift_series(record):
	epochs = record.get("epochs", [])
	if len(epochs) < 2:
		return [float("nan")]
	drift = [float("nan")]
	for idx in range(1, len(epochs)):
		curr = epochs[idx]["potents"]
		prev = epochs[idx - 1]["potents"]
		deltas = []
		for potent, previous in zip(curr, prev):
			mu_curr = np.array(potent["means"], dtype=float)
			mu_prev = np.array(previous["means"], dtype=float)
			sigma_curr = np.array(potent["stds"], dtype=float)
			sigma_prev = np.array(previous["stds"], dtype=float)
			deltas.append(float(np.sqrt(np.sum((mu_curr - mu_prev) ** 2) + np.sum((sigma_curr - sigma_prev) ** 2))))
		drift.append(float(np.mean(deltas)) if deltas else float("nan"))
	return drift


def _support_series(record):
	visualization = record.get("visualization") or {}
	return [
		item.get("kappa_total", np.nan) if isinstance(item, dict) else np.nan
		for item in visualization.get("effective_support", [])
	]


def _usage_series(record):
	return (record.get("visualization") or {}).get("usage_entropy", [])


def _nc1_series(record):
	return (record.get("visualization") or {}).get("nc1", [])


def _plasticity_series(record):
	epochs = record.get("epochs", [])
	if not epochs:
		return []
	values = [1.0]
	for epoch in epochs[1:]:
		potent_values = [
			float(potent["plasticity"]) if potent.get("plasticity") is not None else 1.0
			for potent in epoch.get("potents", [])
			if potent.get("type") != "Neutral"
		]
		values.append(float(np.mean(potent_values)) if potent_values else float("nan"))
	return values


def _differentiation_matrix(record):
	matrices = (record.get("visualization") or {}).get("differentiation_matrices", [])
	for matrix in reversed(matrices):
		if matrix is not None:
			return matrix
	return None


def _potent_mu_matrix(epoch_record):
	rows = []
	for potent in epoch_record.get("potents", []):
		if potent["type"] == "Neutral":
			continue
		rows.append(np.array(potent["means"], dtype=float))
	return np.array(rows, dtype=float) if rows else np.empty((0, 0), dtype=float)


def _linear_cka(x, y, eps=1e-12):
	if x.size == 0 or y.size == 0 or x.shape != y.shape:
		return float("nan")
	xty = y.T @ x
	xtx = x.T @ x
	yty = y.T @ y
	den = np.linalg.norm(xtx, ord="fro") * np.linalg.norm(yty, ord="fro")
	if den <= eps:
		return float("nan")
	return float((np.linalg.norm(xty, ord="fro") ** 2) / den)


def _temporal_cka_series(record):
	epochs = record.get("epochs", [])
	if len(epochs) < 2:
		return [float("nan")]
	reference = _potent_mu_matrix(epochs[-1])
	values = [float("nan")]
	for epoch_record in epochs[1:]:
		values.append(_linear_cka(_potent_mu_matrix(epoch_record), reference))
	return values


def _trajectory_from_record(record):
	trajectory = []
	types = None
	for epoch_record in record.get("epochs", [])[1:]:
		potents = []
		potent_types = []
		for potent in epoch_record.get("potents", []):
			if potent.get("type") == "Neutral":
				continue
			potents.append(np.array(potent["means"], dtype=float))
			potent_types.append(potent["type"])
		if not potents:
			continue
		trajectory.append(np.array(potents, dtype=float))
		types = potent_types
	return np.array(trajectory, dtype=float), types or []


def _movement_from_record(record):
	history = []
	for epoch_record in record.get("epochs", [])[1:]:
		potents = []
		for potent in epoch_record.get("potents", []):
			if potent.get("type") == "Neutral":
				continue
			potents.append(np.concatenate([
				np.array(potent["means"], dtype=float),
				np.array(potent["stds"], dtype=float),
			]))
		if potents:
			history.append(np.array(potents, dtype=float))
	if len(history) < 2:
		return np.empty((0, 0), dtype=float)
	rows = []
	for previous, current in zip(history[:-1], history[1:]):
		rows.append(np.linalg.norm(current - previous, axis=1))
	return np.array(rows, dtype=float)


def _seed42_record(records):
	for record in records:
		if int(record.get("seed", -1)) == 42:
			return record
	return records[0] if records else None


def _differentiation_matrices(record):
	visualization = record.get("visualization") or {}
	return visualization.get("differentiation_matrices", [])


def _model_kwargs_from_trace(record):
	field = record["configuration"]["field"]
	num_classes = int(record["num_classes"])
	num_potents_per_class = int(field["positive_potents"]) // max(num_classes, 1)
	return {
		"embedding-dim": int(field["embedding_dim"]),
		"hidden-dim": int(record.get("configuration", {}).get("model", {}).get("hidden_dim", 512)),
		"neutral-potents": int(field["neutral_potents"]) // max(num_potents_per_class, 1),
		"num-potents-per-class": int(num_potents_per_class),
		"latent-dim": int(field["latent_dim"]),
		"negative-potents": int(field["negative_potents"]) > 0,
		"num-classes": num_classes,
		"plasticity": any("plasticity" in potent and potent["plasticity"] is not None for potent in record["epochs"][-1].get("potents", [])),
	}


def _instantiate_image_model(dataset_name, record):
	args = _model_kwargs_from_trace(record)
	if dataset_name in {"MNIST", "FMNIST"}:
		return instantiate_mnist_accommodation_classifier(args)
	if dataset_name == "CIFAR10":
		return instantiate_cifar_accommodation_classifier(args)
	raise ValueError(f"Unsupported image dataset for semantics: {dataset_name}")


def _dataset_vocab_size(dataset_name, data_dir):
	data_dir = Path(data_dir)
	if dataset_name in {"AGNEWS", "AGNews"}:
		return AGNewsDataset(str(data_dir)).vocab_size
	if dataset_name == "IMDB":
		return IMDBDataset(str(data_dir)).vocab_size
	if dataset_name == "YELP":
		return YelpReviewsStarsDataset(str(data_dir)).vocab_size
	raise ValueError(f"Unsupported text dataset: {dataset_name}")


def _instantiate_model_for_dataset(dataset_name, record, data_dir):
	args = _model_kwargs_from_trace(record)
	if dataset_name in IMAGE_DATASETS:
		return _instantiate_image_model(dataset_name, record)
	if dataset_name in {"AGNEWS", "AGNews", "IMDB", "YELP"}:
		return instantiate_gru_accommodation_classifier(args, _dataset_vocab_size(dataset_name, data_dir))
	raise ValueError(f"Unsupported dataset for model cache: {dataset_name}")


def _display_image_from_batch(dataset_name, image_tensor):
	image = image_tensor.detach().cpu().numpy()
	if dataset_name in {"MNIST", "FMNIST"}:
		image = 1.0 - image
		image = np.clip(image.squeeze(), 0.0, 1.0)
		return image
	if dataset_name == "CIFAR10":
		mean = np.array([0.4914, 0.4822, 0.4465], dtype=float).reshape(3, 1, 1)
		std = np.array([0.2470, 0.2435, 0.2616], dtype=float).reshape(3, 1, 1)
		image = np.clip(image * std + mean, 0.0, 1.0)
		return np.transpose(image, (1, 2, 0))
	return image


def _build_image_val_loader(dataset_name, data_dir, batch_size=256, device=None):
	data_dir = Path(data_dir)
	device = _resolve_device(device)
	if dataset_name == "MNIST":
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,)),
		])
		dataset = MNISTDataset(root=str(data_dir), train=False, transform=transform)
	elif dataset_name in {"FMNIST"}:
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,)),
		])
		dataset = FashionMNISTDataset(root=str(data_dir), train=False, transform=transform)
	elif dataset_name == "CIFAR10":
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
		])
		dataset = Cifar10Dataset(root=str(data_dir), train=False, transform=transform)
	else:
		raise ValueError(f"Unsupported image dataset for semantics: {dataset_name}")
	return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=(device == "cuda"))


def _build_validation_loader(dataset_name, data_dir, device=None):
	data_dir = Path(data_dir)
	device = _resolve_device(device)
	if dataset_name in IMAGE_DATASETS:
		return _build_image_val_loader(dataset_name, data_dir=data_dir, batch_size=256, device=device)
	if dataset_name in {"AGNEWS", "AGNews"}:
		dataset = AGNewsDataset(str(data_dir))
		train_size = int(0.8 * len(dataset))
		val_size = len(dataset) - train_size
		_, val_dataset = random_split(
			dataset,
			[train_size, val_size],
			generator=torch.Generator().manual_seed(42),
		)
		return DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False, pin_memory=(device == "cuda"))
	if dataset_name == "IMDB":
		dataset = IMDBDataset(str(data_dir))
		train_size = int(0.8 * len(dataset))
		val_size = len(dataset) - train_size
		_, val_dataset = random_split(
			dataset,
			[train_size, val_size],
			generator=torch.Generator().manual_seed(42),
		)
		return DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False, pin_memory=(device == "cuda"))
	if dataset_name == "YELP":
		dataset = YelpReviewsStarsDataset(str(data_dir))
		train_size = int(0.8 * len(dataset))
		val_size = len(dataset) - train_size
		_, val_dataset = random_split(
			dataset,
			[train_size, val_size],
			generator=torch.Generator().manual_seed(42),
		)
		return DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False, pin_memory=(device == "cuda"))
	raise ValueError(f"Unsupported dataset for validation loader: {dataset_name}")


def _snapshot_path_for_record(config_dir: Path, experiment_type: str, dataset_name: str, record):
	field = record["configuration"]["field"]
	seed = int(record["seed"])
	pospot = int(field["positive_potents"])
	neutral = int(field["neutral_potents"])
	negpot = int(field["negative_potents"])
	epoch = len(record.get("epochs", [])) - 1
	return (
		config_dir
		/ experiment_type
		/ dataset_name
		/ "Likelihood"
		/ f"POS{pospot}-NEU{neutral}-NEG{negpot}"
		/ str(seed)
		/ f"snapshot_{epoch:02d}.pt"
	)


def _snapshot_dir_for_record(config_dir: Path, experiment_type: str, dataset_name: str, record):
	field = record["configuration"]["field"]
	seed = int(record["seed"])
	pospot = int(field["positive_potents"])
	neutral = int(field["neutral_potents"])
	negpot = int(field["negative_potents"])
	return (
		config_dir
		/ experiment_type
		/ dataset_name
		/ "Likelihood"
		/ f"POS{pospot}-NEU{neutral}-NEG{negpot}"
		/ str(seed)
	)


def _snapshot_epochs_for_record(config_dir: Path, experiment_type: str, dataset_name: str, record):
	snapshot_dir = _snapshot_dir_for_record(config_dir, experiment_type, dataset_name, record)
	epochs = []
	for path in sorted(snapshot_dir.glob("snapshot_*.pt")):
		try:
			epochs.append(int(path.stem.split("_")[-1]))
		except ValueError:
			continue
	return epochs


def _extract_all_potents(model):
	layer = model.accommodation_layer
	potents = []
	meta = []
	for c in range(layer.positive_mu.shape[0]):
		for k in range(layer.positive_mu.shape[1]):
			potents.append((layer.positive_mu[c, k], layer.positive_sigma[c, k]))
			meta.append((c, k, "Positive"))
	if layer.negative_mu is not None:
		for c in range(layer.negative_mu.shape[0]):
			for k in range(layer.negative_mu.shape[1]):
				potents.append((layer.negative_mu[c, k], layer.negative_sigma[c, k]))
				meta.append((c, k, "Negative"))
	return potents, meta


def _compute_potent_usage_from_validation(model, val_loader, potents, device):
	all_scores = [[] for _ in potents]
	with torch.no_grad():
		for x, _ in val_loader:
			x = x.to(device)
			features = model.backbone(x)
			x_mu = model.accommodation_layer.mu_encoder(features).unsqueeze(1).unsqueeze(2)
			x_sigma = model.accommodation_layer.sigma_encoder(features).unsqueeze(1).unsqueeze(2)
			for idx, (mu, sigma) in enumerate(potents):
				similarity = compatibility_operator(
					x_mu,
					x_sigma,
					mu.unsqueeze(0).unsqueeze(0).unsqueeze(0),
					sigma.unsqueeze(0).unsqueeze(0).unsqueeze(0),
				)
				score = similarity.squeeze()
				if score.dim() == 0:
					score = score.unsqueeze(0)
				score = score.view(score.shape[0], -1).max(dim=1).values.detach().cpu().numpy()
				all_scores[idx].extend(score.tolist())
	return np.array([
		float(np.mean(scores)) if scores else float("-inf")
		for scores in all_scores
	], dtype=float)


def _extract_final_model_cache(dataset_name, record, snapshot_path, data_dir, device=None):
	if not snapshot_path.exists():
		return None

	device = _resolve_device(device)
	model = _instantiate_model_for_dataset(dataset_name, record, data_dir).to(device)
	state_dict = torch.load(snapshot_path, map_location=device)
	model.load_state_dict(state_dict)
	model.eval()

	layer = model.accommodation_layer
	potents = []

	def _append_group(mu_tensor, sigma_tensor, plasticity_tensor, potent_type):
		if mu_tensor is None:
			return
		mu = mu_tensor.detach().cpu().numpy()
		sigma = sigma_tensor.detach().cpu().numpy()
		plasticity = None if plasticity_tensor is None else plasticity_tensor.detach().cpu().numpy()
		for clazz in range(mu.shape[0]):
			for potent_idx in range(mu.shape[1]):
				potents.append({
					"class": int(clazz),
					"potent_idx": int(potent_idx),
					"type": potent_type,
					"means": mu[clazz, potent_idx],
					"stds": sigma[clazz, potent_idx],
					"plasticity": None if plasticity is None else plasticity[clazz, potent_idx],
				})

	_append_group(
		layer.positive_mu,
		layer.positive_sigma,
		getattr(layer, "positive_plasticity", None),
		"Positive",
	)
	_append_group(
		layer.negative_mu,
		layer.negative_sigma,
		getattr(layer, "negative_plasticity", None),
		"Negative",
	)

	return {
		"seed": int(record["seed"]),
		"epoch": len(record.get("epochs", [])) - 1,
		"potents": potents,
		"positive_usage": None if getattr(layer, "positive_usage", None) is None else layer.positive_usage.detach().cpu().numpy(),
		"negative_usage": None if getattr(layer, "negative_usage", None) is None else layer.negative_usage.detach().cpu().numpy(),
	}


def _recompute_visualization_from_snapshots(dataset_name, record, config_dir, experiment_type, data_dir, device=None, log_prefix=""):
	snapshot_epochs = _snapshot_epochs_for_record(config_dir, experiment_type, dataset_name, record)
	if not snapshot_epochs:
		return None

	device = _resolve_device(device)
	val_loader = _build_validation_loader(dataset_name, data_dir=data_dir, device=device)
	model = _instantiate_model_for_dataset(dataset_name, record, data_dir).to(device)

	effective_support = [None]
	usage_entropy = [None]
	nc1 = [None]
	nc1_components = [None]
	differentiation_matrices = [None]

	for epoch in snapshot_epochs:
		print(f"{log_prefix}recomputing epoch {epoch:02d}/{snapshot_epochs[-1]:02d} on {device}")
		snapshot_path = _snapshot_path_for_record(config_dir, experiment_type, dataset_name, {
			**record,
			"epochs": [None] * (epoch + 1),
		})
		if not snapshot_path.exists():
			continue
		state_dict = torch.load(snapshot_path, map_location=device)
		model.load_state_dict(state_dict)
		model.eval()
		stats = compute_epoch_visualization_stats(model, val_loader, device)
		while len(effective_support) < epoch:
			effective_support.append(None)
			usage_entropy.append(None)
			nc1.append(None)
			nc1_components.append(None)
			differentiation_matrices.append(None)
		effective_support.append(stats["effective_support"])
		usage_entropy.append(stats["usage_entropy"])
		nc1.append(stats["nc1"])
		nc1_components.append(stats["nc1_components"])
		differentiation_matrices.append(stats["differentiation_matrix"])

	return {
		"effective_support": effective_support,
		"usage_entropy": usage_entropy,
		"nc1": nc1,
		"nc1_components": nc1_components,
		"differentiation_matrices": differentiation_matrices,
	}


def _compute_semantics_payload(dataset_name, record, snapshot_path, data_dir, top_k=20, num_potents=16, batch_size=256, device=None):
	if dataset_name not in IMAGE_DATASETS or not snapshot_path.exists():
		return None

	device = _resolve_device(device)
	model = _instantiate_image_model(dataset_name, record).to(device)
	state_dict = torch.load(snapshot_path, map_location=device)
	model.load_state_dict(state_dict)
	model.eval()

	val_loader = _build_image_val_loader(dataset_name, data_dir=data_dir, batch_size=batch_size, device=device)
	potents, meta = _extract_all_potents(model)
	usage_pos_tensor = getattr(model.accommodation_layer, "positive_usage", None)
	usage_neg_tensor = getattr(model.accommodation_layer, "negative_usage", None)
	if usage_pos_tensor is not None:
		usage_pos = usage_pos_tensor.detach().cpu().numpy().flatten()
		usage_neg = (
			usage_neg_tensor.detach().cpu().numpy().flatten()
			if usage_neg_tensor is not None else np.array([], dtype=float)
		)
		usage = np.concatenate([usage_pos, usage_neg])
	else:
		# Older or non-plastic models do not track usage, so estimate it from validation compatibility.
		usage = _compute_potent_usage_from_validation(model, val_loader, potents, device)
	top_idx = np.argsort(usage)[-min(num_potents, len(usage)):].tolist()
	all_scores = {idx: [] for idx in top_idx}

	with torch.no_grad():
		for x, _ in val_loader:
			x = x.to(device)
			features = model.backbone(x)
			x_mu = model.accommodation_layer.mu_encoder(features).unsqueeze(1).unsqueeze(2)
			x_sigma = model.accommodation_layer.sigma_encoder(features).unsqueeze(1).unsqueeze(2)
			for idx in top_idx:
				mu, sigma = potents[idx]
				similarity = compatibility_operator(
					x_mu,
					x_sigma,
					mu.unsqueeze(0).unsqueeze(0).unsqueeze(0),
					sigma.unsqueeze(0).unsqueeze(0).unsqueeze(0),
				)
				score = similarity.squeeze()
				if score.dim() == 0:
					score = score.unsqueeze(0)
				score = score.view(score.shape[0], -1).max(dim=1).values.detach().cpu().numpy()
				for sample_idx, score_value in enumerate(score):
					all_scores[idx].append((_display_image_from_batch(dataset_name, x[sample_idx]), float(score_value)))

	top_images = {}
	for idx in top_idx:
		sorted_images = sorted(all_scores[idx], key=lambda item: item[1], reverse=True)
		top_images[str(idx)] = [image for image, _ in sorted_images[:top_k]]

	return {
		"seed": int(record["seed"]),
		"epoch": len(record.get("epochs", [])) - 1,
		"top_idx": top_idx,
		"meta": meta,
		"top_images": top_images,
		"top_k": top_k,
		"num_potents": num_potents,
	}


def _series_matrix(rows):
	if not rows:
		return np.empty((0, 0), dtype=float)
	max_len = max(len(row) for row in rows)
	matrix = np.full((len(rows), max_len), np.nan, dtype=float)
	for idx, row in enumerate(rows):
		row_arr = np.asarray(row, dtype=float)
		matrix[idx, : len(row_arr)] = row_arr
	return matrix


def _flatten_pairs(x_matrix, y_matrix):
	if x_matrix.size == 0 or y_matrix.size == 0:
		return np.array([]), np.array([])
	length = min(x_matrix.shape[1], y_matrix.shape[1])
	x = x_matrix[:, :length].reshape(-1)
	y = y_matrix[:, :length].reshape(-1)
	mask = np.isfinite(x) & np.isfinite(y)
	return x[mask], y[mask]


def _cache_path(cache_dir: Path, dataset: str, experiment_type: str, config_name: str):
	return cache_dir / dataset / experiment_type / f"{config_name}.json"


def list_result_groups(results_dir="results", dataset=None, types=("accommodation", "frozen-backbone")):
	results_dir = Path(results_dir)
	if not results_dir.exists():
		return []

	groups = []
	for experiment_type in types:
		grouped = {}
		for path in _trace_json_paths(results_dir, experiment_type, dataset=dataset):
			dataset_name = path.parent.name
			config_name = path.parents[2].name
			grouped.setdefault((dataset_name, config_name), []).append(path)
		for (dataset_name, config_name), paths in sorted(grouped.items()):
			groups.append({
				"dataset": dataset_name,
				"type": experiment_type,
				"config_name": config_name,
				"paths": paths,
				"config_dir": paths[0].parents[2] if paths else None,
			})
	return groups


def build_visualization_cache(
	results_dir="results",
	cache_dir="visualization_cache",
	data_dir="data",
	dataset=None,
	types=("accommodation", "frozen-backbone"),
	include_rich_cache=True,
	device=None,
):
	results_dir = Path(results_dir)
	cache_dir = Path(cache_dir)
	device = _resolve_device(device)
	if not results_dir.exists():
		print(f"Results directory not found: {results_dir}")
		return []

	written = []
	print(f"Building visualization cache on {device}")
	for group in list_result_groups(results_dir=results_dir, dataset=dataset, types=types):
		print(f"[{group['dataset']}] [{group['type']}] [{group['config_name']}] loading traces")
		records = []
		for path_index, path in enumerate(group["paths"], start=1):
			print(f"[{group['dataset']}] [{group['type']}] [{group['config_name']}] trace {path_index}/{len(group['paths'])}: {path.name}")
			try:
				record = _load_json(path)
			except (OSError, json.JSONDecodeError):
				print(f"[{group['dataset']}] [{group['type']}] [{group['config_name']}] failed to read {path.name}")
				continue
			if include_rich_cache and record.get("visualization") is None:
				print(f"[{group['dataset']}] [{group['type']}] [{group['config_name']}] seed={record.get('seed')} missing embedded visualization, recomputing from snapshots")
				try:
					record["visualization"] = _recompute_visualization_from_snapshots(
						group["dataset"],
						record,
						group["config_dir"],
						group["type"],
						data_dir=data_dir,
						device=device,
						log_prefix=f"[{group['dataset']}] [{group['type']}] [{group['config_name']}] seed={record.get('seed')} | ",
					)
				except Exception:
					record["visualization"] = None
					print(f"[{group['dataset']}] [{group['type']}] [{group['config_name']}] seed={record.get('seed')} failed to recompute from snapshots")
			else:
				print(f"[{group['dataset']}] [{group['type']}] [{group['config_name']}] seed={record.get('seed')} using embedded visualization")
			records.append(record)
		if not records:
			print(f"[{group['dataset']}] [{group['type']}] [{group['config_name']}] no readable records")
			continue
		seed42_record = _seed42_record(records)
		semantics_payload = None
		model_payload = None
		snapshot_path = None
		if include_rich_cache and seed42_record is not None:
			snapshot_path = _snapshot_path_for_record(group["config_dir"], group["type"], group["dataset"], seed42_record)
			print(f"[{group['dataset']}] [{group['type']}] [{group['config_name']}] building final model cache from seed={seed42_record.get('seed')} on {device}")
			try:
				model_payload = _extract_final_model_cache(
					group["dataset"],
					seed42_record,
					snapshot_path,
					data_dir=data_dir,
					device=device,
				)
			except Exception:
				model_payload = None
				print(f"[{group['dataset']}] [{group['type']}] [{group['config_name']}] failed building final model cache")
		if include_rich_cache and seed42_record is not None and group["dataset"] in IMAGE_DATASETS:
			print(f"[{group['dataset']}] [{group['type']}] [{group['config_name']}] building semantics from seed={seed42_record.get('seed')} on {device}")
			try:
				semantics_payload = _compute_semantics_payload(
					group["dataset"],
					seed42_record,
					snapshot_path,
					data_dir=data_dir,
					device=device,
				)
			except Exception:
				semantics_payload = None
				print(f"[{group['dataset']}] [{group['type']}] [{group['config_name']}] failed building semantics")
		seed42_trajectory = None
		if seed42_record is not None:
			trajectory_values, trajectory_types = _trajectory_from_record(seed42_record)
			seed42_trajectory = {
				"seed": int(seed42_record["seed"]),
				"movement": _movement_from_record(seed42_record),
				"trajectory": trajectory_values,
				"types": trajectory_types,
			}
		payload = {
			"dataset": group["dataset"],
			"type": group["type"],
			"config_name": group["config_name"],
			"drift": [_calculate_drift_series(record) for record in records],
			"support": [_support_series(record) for record in records if record.get("visualization") is not None],
			"usage": [_usage_series(record) for record in records if record.get("visualization") is not None],
			"nc1": [_nc1_series(record) for record in records if record.get("visualization") is not None],
			"plasticity": [_plasticity_series(record) for record in records],
			"cka": [_temporal_cka_series(record) for record in records],
			"final_differentiation_matrix": _differentiation_matrix(records[0]),
			"differentiation_matrices": _differentiation_matrices(seed42_record) if seed42_record is not None else [],
			"seed42_trajectory": seed42_trajectory,
			"seed42_model_cache": model_payload,
			"seed42_semantics": semantics_payload,
			"num_records": len(records),
		}
		path = _cache_path(cache_dir, group["dataset"], group["type"], group["config_name"])
		_write_json(path, _json_ready(payload))
		print(f"[{group['dataset']}] [{group['type']}] [{group['config_name']}] wrote cache -> {path}")
		written.append(path)

	return written



def dataset_cache_coverage(results_dir="results", cache_dir="visualization_cache", dataset=None, types=("accommodation", "frozen-backbone")):
	results_dir = Path(results_dir)
	cache_dir = Path(cache_dir)
	coverage = []
	for group in list_result_groups(results_dir=results_dir, dataset=dataset, types=types):
		cache_path = _cache_path(cache_dir, group["dataset"], group["type"], group["config_name"])
		expected = 0
		for path in group["paths"]:
			try:
				_load_json(path)
			except (OSError, json.JSONDecodeError):
				continue
			expected += 1

		actual = 0
		if cache_path.exists():
			try:
				actual = int(_load_json(cache_path).get("num_records", 0))
			except (OSError, json.JSONDecodeError, ValueError, TypeError):
				actual = 0

		coverage.append({
			"dataset": group["dataset"],
			"type": group["type"],
			"config_name": group["config_name"],
			"expected_records": expected,
			"cached_records": actual,
			"cache_path": cache_path,
			"is_complete": actual >= expected and expected > 0,
		})
	return coverage


def _plot_mean_std(ax, matrix, title, color):
	ax.set_title(title)
	ax.set_xlabel("Epoch")
	ax.grid(alpha=0.3)
	if matrix.size == 0:
		ax.text(0.5, 0.5, "No cache", ha="center", va="center", transform=ax.transAxes)
		return
	x = np.arange(matrix.shape[1])
	mean = np.nanmean(matrix, axis=0)
	std = np.nanstd(matrix, axis=0)
	ax.plot(x, mean, color=color, linewidth=2.4)
	ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)


def _plot_scatter(ax, xs, ys, title, xlabel, color):
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel("NC1")
	ax.grid(alpha=0.3)
	if xs.size == 0 or ys.size == 0:
		ax.text(0.5, 0.5, "No cache", ha="center", va="center", transform=ax.transAxes)
		return
	ax.scatter(xs, ys, s=10, alpha=0.35, color=color, edgecolors="none")


def _save_figure_by_format(fig, dataset_output_dir: Path, stem: str):
	written = []
	save_specs = (
		("png", {"dpi": 180, "bbox_inches": "tight", "facecolor": "white"}),
		("pdf", {"bbox_inches": "tight", "facecolor": "white"}),
		("svg", {"bbox_inches": "tight", "facecolor": "white"}),
	)
	for extension, kwargs in save_specs:
		format_dir = dataset_output_dir / extension
		format_dir.mkdir(parents=True, exist_ok=True)
		path = format_dir / f"{stem}.{extension}"
		fig.savefig(path, **kwargs)
		written.append(path)
	return written


def _build_config_figure(cache_payload, output_dir: Path):
	dataset = cache_payload["dataset"]
	experiment_type = cache_payload["type"]
	config_name = cache_payload["config_name"]
	drift = _series_matrix(cache_payload["drift"])
	support = _series_matrix(cache_payload["support"])
	usage = _series_matrix(cache_payload["usage"])
	nc1 = _series_matrix(cache_payload["nc1"])
	cka = _series_matrix(cache_payload["cka"])
	matrix = None if cache_payload["final_differentiation_matrix"] is None else np.array(cache_payload["final_differentiation_matrix"], dtype=float)

	support_x, support_y = _flatten_pairs(support, nc1)
	usage_x, usage_y = _flatten_pairs(usage, nc1)
	drift_x, drift_y = _flatten_pairs(drift, nc1)

	fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
	fig.suptitle(f"{dataset} | {experiment_type} | {config_name}", fontsize=16)

	_plot_mean_std(axes[0, 0], drift, "Drift", "tab:blue")
	_plot_mean_std(axes[0, 1], support, "Effective Support", "tab:green")
	_plot_mean_std(axes[0, 2], usage, "Effective number of potents", "tab:orange")
	_plot_mean_std(axes[1, 0], nc1, "NC1", "tab:red")
	_plot_mean_std(axes[1, 1], cka, "Temporal potent CKA", "tab:purple")

	axes[1, 2].set_title("Final differentiation matrix")
	if matrix is None:
		axes[1, 2].text(0.5, 0.5, "No cache", ha="center", va="center", transform=axes[1, 2].transAxes)
	else:
		im = axes[1, 2].imshow(matrix, cmap="Blues", aspect="auto")
		fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
	axes[1, 2].set_xticks([])
	axes[1, 2].set_yticks([])

	_plot_scatter(axes[2, 0], support_x, support_y, "Effective Support vs NC1", "Effective Support", "tab:green")
	_plot_scatter(axes[2, 1], usage_x, usage_y, "Effective number of potents vs NC1", "Effective number of potents", "tab:orange")
	_plot_scatter(axes[2, 2], drift_x, drift_y, "Drift vs NC1", "Drift", "tab:blue")

	stem = f"{dataset}__{experiment_type}__{config_name}"
	written = _save_figure_by_format(fig, output_dir, stem)
	plt.close(fig)
	return written


def _pca_2d(matrix):
	matrix = np.asarray(matrix, dtype=float)
	if matrix.ndim != 2 or matrix.shape[0] == 0:
		return np.empty((0, 2), dtype=float)
	matrix = matrix - matrix.mean(axis=0, keepdims=True)
	_, _, vt = np.linalg.svd(matrix, full_matrices=False)
	components = vt[:2].T
	return matrix @ components


def _build_trajectory_figure(cache_payload, output_dir: Path):
	trajectory_payload = cache_payload.get("seed42_trajectory")
	if not trajectory_payload:
		return []

	trajectory = np.array(trajectory_payload.get("trajectory", []), dtype=float)
	movement = np.array(trajectory_payload.get("movement", []), dtype=float)
	types = np.array(trajectory_payload.get("types", []))
	if trajectory.size == 0 or len(types) == 0:
		return []

	flat = trajectory.reshape(-1, trajectory.shape[-1])
	trajectory_2d = _pca_2d(flat).reshape(trajectory.shape[0], trajectory.shape[1], 2)
	color_map = {"Positive": "green", "Negative": "red"}
	y_max = float(np.nanmax(movement)) if movement.size else 1.0

	fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
	for potent_idx in range(trajectory_2d.shape[1]):
		path = trajectory_2d[:, potent_idx]
		axes[0].plot(path[:, 0], path[:, 1], color=color_map.get(types[potent_idx], "gray"), alpha=0.6, linewidth=0.8)
	axes[0].set_title(f"Potent trajectories (seed={trajectory_payload.get('seed', 42)})")
	axes[0].set_xticks([])
	axes[0].set_yticks([])

	if movement.size:
		epochs = np.arange(1, movement.shape[0] + 1)
		for potent_type, color in color_map.items():
			mask = types == potent_type
			if np.sum(mask) == 0:
				continue
			axes[1].plot(epochs, movement[:, mask].mean(axis=1), label=potent_type, color=color, linewidth=2)
		axes[1].set_ylim(0.0, y_max * 1.05 if y_max > 0 else 1.0)
	axes[1].set_title("Per-potent drift")
	axes[1].set_xlabel("Epoch")
	axes[1].grid(alpha=0.3)
	axes[1].legend(frameon=False)

	stem = f"{cache_payload['dataset']}__{cache_payload['type']}__{cache_payload['config_name']}__trajectory"
	written = _save_figure_by_format(fig, output_dir, stem)
	plt.close(fig)
	return written


def _build_semantics_figure(cache_payload, output_dir: Path):
	semantics = cache_payload.get("seed42_semantics")
	if not semantics:
		return []

	top_idx = semantics.get("top_idx", [])
	top_k = int(semantics.get("top_k", 0))
	if not top_idx or top_k <= 0:
		return []

	fig, axes = plt.subplots(len(top_idx), top_k, figsize=(top_k * 0.6, max(len(top_idx), 1) * 0.6))
	axes = np.atleast_2d(axes)
	meta = [tuple(item) for item in semantics.get("meta", [])]
	top_images = {int(key): value for key, value in semantics.get("top_images", {}).items()}
	for row, idx in enumerate(top_idx):
		clazz, potent_idx, potent_type = meta[idx]
		polarity = "+" if potent_type == "Positive" else "-"
		for col in range(top_k):
			ax = axes[row, col]
			image = np.array(top_images[idx][col], dtype=float)
			if image.ndim == 2:
				ax.imshow(image, cmap="gray", interpolation="nearest")
			else:
				ax.imshow(image, interpolation="nearest")
			ax.axis("off")
			if col == 0:
				ax.text(-0.2, 0.5, f"{polarity}[{clazz}]_{potent_idx}", transform=ax.transAxes, ha="right", va="center", fontsize=9)

	fig.subplots_adjust(left=0.08, right=0.995, top=0.98, bottom=0.02, wspace=0.01, hspace=0.01)
	stem = f"{cache_payload['dataset']}__{cache_payload['type']}__{cache_payload['config_name']}__semantics"
	written = _save_figure_by_format(fig, output_dir, stem)
	plt.close(fig)
	return written


def _build_differentiation_epochs_figure(cache_payload, output_dir: Path):
	matrices = cache_payload.get("differentiation_matrices") or []
	valid = [(idx, np.array(matrix, dtype=float)) for idx, matrix in enumerate(matrices) if matrix is not None]
	if not valid:
		return []

	target_indices = []
	for candidate in (1, 40, len(matrices) - 1):
		if 0 <= candidate < len(matrices) and matrices[candidate] is not None and candidate not in target_indices:
			target_indices.append(candidate)
	if not target_indices:
		target_indices = [valid[0][0], valid[-1][0]]

	selected = [np.array(matrices[idx], dtype=float) for idx in target_indices]
	vmin = min(float(np.min(matrix)) for matrix in selected)
	vmax = max(float(np.max(matrix)) for matrix in selected)
	fig, axes = plt.subplots(1, len(selected), figsize=(4.5 * len(selected), 4), constrained_layout=True)
	axes = np.atleast_1d(axes)
	for ax, idx, matrix in zip(axes, target_indices, selected):
		im = ax.imshow(matrix, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
		ax.set_title(f"Epoch {idx}")
		ax.set_xticks([])
		ax.set_yticks([])
	fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04)

	stem = f"{cache_payload['dataset']}__{cache_payload['type']}__{cache_payload['config_name']}__differentiation_epochs"
	written = _save_figure_by_format(fig, output_dir, stem)
	plt.close(fig)
	return written


def render_dataset_visualizations_from_cache(cache_dir="visualization_cache", output_dir="dataset_visualizations", dataset=None, types=("accommodation", "frozen-backbone")):
	cache_dir = Path(cache_dir)
	output_root = Path(output_dir)
	output_root.mkdir(parents=True, exist_ok=True)

	if not cache_dir.exists():
		print(f"Visualization cache directory not found: {cache_dir}")
		return []

	written = []
	datasets = [dataset] if dataset is not None else sorted(path.name for path in cache_dir.iterdir() if path.is_dir())
	for dataset_name in datasets:
		for experiment_type in types:
			type_dir = cache_dir / dataset_name / experiment_type
			if not type_dir.exists():
				continue
			for cache_path in sorted(type_dir.glob("*.json")):
				payload = _load_json(cache_path)
				output_dir_for_type = output_root / dataset_name
				written.extend(_build_config_figure(payload, output_dir_for_type))
				written.extend(_build_trajectory_figure(payload, output_dir_for_type))
				written.extend(_build_differentiation_epochs_figure(payload, output_dir_for_type))
				written.extend(_build_semantics_figure(payload, output_dir_for_type))
	return written


generate_all_dataset_visualizations = render_dataset_visualizations_from_cache
build_dataset_visualization_cache = build_visualization_cache
dataset_visualization_cache_coverage = dataset_cache_coverage
