import numpy as np
from scipy.stats import pearsonr


def calculate_nc1_details(embeddings, labels, eps=1e-12):
	embeddings = np.asarray(embeddings, dtype=float)
	labels = np.asarray(labels)
	if embeddings.ndim != 2:
		raise ValueError(f"embeddings must be a 2D array, got shape {embeddings.shape}")
	if labels.ndim != 1:
		raise ValueError(f"labels must be a 1D array, got shape {labels.shape}")
	if embeddings.shape[0] != labels.shape[0]:
		raise ValueError("embeddings and labels must contain the same number of samples")

	classes = np.unique(labels)
	num_classes = len(classes)
	if num_classes < 2:
		return {
			"within_trace": float("nan"),
			"between_trace": float("nan"),
			"sigma_w_trace": float("nan"),
			"sigma_b_trace": float("nan"),
			"nc1": float("nan"),
		}

	num_samples, latent_dim = embeddings.shape
	global_mean = embeddings.mean(axis=0)
	sigma_w = np.zeros((latent_dim, latent_dim), dtype=float)
	sigma_b = np.zeros((latent_dim, latent_dim), dtype=float)
	trace_within = 0.0
	trace_between = 0.0

	for clazz in classes:
		class_embeddings = embeddings[labels == clazz]
		if class_embeddings.size == 0:
			continue
		class_mean = class_embeddings.mean(axis=0)
		centered = class_embeddings - class_mean
		trace_within += float(np.sum(centered * centered))
		sigma_w += centered.T @ centered
		diff = class_mean - global_mean
		trace_between += float(np.dot(diff, diff))
		sigma_b += np.outer(diff, diff)

	sigma_w /= max(num_samples, 1)
	sigma_b /= num_classes
	sigma_b_pinv = np.linalg.pinv(sigma_b, rcond=eps)
	nc1 = float(np.trace(sigma_w @ sigma_b_pinv) / num_classes)
	return {
		"within_trace": trace_within,
		"between_trace": trace_between,
		"sigma_w_trace": float(np.trace(sigma_w)),
		"sigma_b_trace": float(np.trace(sigma_b)),
		"nc1": nc1,
	}


def calculate_nc1(embeddings, labels, eps=1e-12):
	details = calculate_nc1_details(embeddings, labels, eps=eps)
	return details["nc1"]


def finite_pearsonr(x, y):
	x = np.asarray(x, dtype=float).reshape(-1)
	y = np.asarray(y, dtype=float).reshape(-1)
	if x.shape != y.shape:
		raise ValueError("x and y must have the same shape")
	mask = np.isfinite(x) & np.isfinite(y)
	if mask.sum() < 2:
		return {"r": float("nan"), "p": float("nan"), "n": int(mask.sum())}
	x = x[mask]
	y = y[mask]
	if np.allclose(x, x[0]) or np.allclose(y, y[0]):
		return {"r": float("nan"), "p": float("nan"), "n": int(mask.sum())}
	r, p = pearsonr(x, y)
	return {"r": float(r), "p": float(p), "n": int(mask.sum())}
