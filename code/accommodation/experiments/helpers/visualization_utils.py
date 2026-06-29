import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import chi2, pearsonr
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from accommodation.datasets.text.agnews import AGNewsDataset
from accommodation.datasets.text.imdb import IMDBDataset
from accommodation.datasets.text.yelp_reviews import YelpReviewsStarsDataset
from accommodation.datasets.vision.cifar10 import Cifar10Dataset
from accommodation.datasets.vision.fashion_mnist import FashionMNISTDataset
from accommodation.datasets.vision.mnist import MNISTDataset
from accommodation.experiments.classifiers.cifar_accommodation_classifier import instantiate_cifar_accommodation_classifier
from accommodation.experiments.classifiers.gru_accommodation_classifier import instantiate_gru_accommodation_classifier
from accommodation.experiments.classifiers.mnist_accommodation_classifier import instantiate_mnist_accommodation_classifier


RUN_GROUPS = {
	"baseline": "PLFalse_GM5_DF0_PPC5",
	"df": {
		"DF-0.5": "PLFalse_GM5_DF0.5_PPC5",
		"DF-0.8": "PLFalse_GM5_DF0.8_PPC5",
		"DF-0.9": "PLFalse_GM5_DF0.9_PPC5",
		"DF-0.95": "PLFalse_GM5_DF0.95_PPC5",
	},
	"plasticity": {
		"PL-1": "PLTrue_GM1_DF0_PPC5",
		"PL-5": "PLTrue_GM5_DF0_PPC5",
		"PL-10": "PLTrue_GM10_DF0_PPC5",
		"PL-20": "PLTrue_GM20_DF0_PPC5",
	},
	"combined": {
		"DF-0.5": "PLTrue_GM5_DF0.5_PPC5",
		"DF-0.8": "PLTrue_GM5_DF0.8_PPC5",
		"DF-0.9": "PLTrue_GM5_DF0.9_PPC5",
		"DF-0.95": "PLTrue_GM5_DF0.95_PPC5",
	},
	"ppc_plasticity": {
		"PPC-1": "PLTrue_GM5_DF0.8_PPC1",
		"PPC-5": "PLTrue_GM5_DF0.8_PPC5",
		"PPC-10": "PLTrue_GM5_DF0.8_PPC10",
		"PPC-20": "PLTrue_GM5_DF0.8_PPC20",
	},
	"ppc_no_plasticity": {
		"PPC-1": "PLFalse_GM5_DF0_PPC1",
		"PPC-5": "PLFalse_GM5_DF0_PPC5",
		"PPC-10": "PLFalse_GM5_DF0_PPC10",
		"PPC-20": "PLFalse_GM5_DF0_PPC20",
	},
}

FROZEN_RUN_GROUPS = {
	"baseline": "PLFalse_GM5_DF0_PPC5_BBE40",
	"combined": "PLTrue_GM5_DF0.8_PPC5_BBE40",
}

DEFAULT_DATASETS = ("MNIST", "FMNIST", "CIFAR10", "AGNEWS", "IMDB", "YELP")
IMAGE_DATASETS = {"MNIST", "FMNIST", "CIFAR10"}
IMAGE_CLASS_NAMES = {
	"MNIST": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
	"FMNIST": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
	"CIFAR10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
}
DATASET_COLORS = {
	"CIFAR10": "#377eb8",
	"IMDB": "#984ea3",
	"YELP": "#1b7f1b",
	"AGNEWS": "#e41a1c",
	"FMNIST": "#4daf4a",
	"MNIST": "#ff7f00",
}
DATASET_SPECS = {
	"MNIST": {"num-classes": 10, "embedding-dim": 128, "hidden-dim": 512, "latent-dim": 30, "input-dim": None},
	"FMNIST": {"num-classes": 10, "embedding-dim": 128, "hidden-dim": 512, "latent-dim": 30, "input-dim": None},
	"CIFAR10": {"num-classes": 10, "embedding-dim": 256, "hidden-dim": 512, "latent-dim": 30, "input-dim": None},
	"AGNEWS": {"num-classes": 4, "embedding-dim": 128, "hidden-dim": 512, "latent-dim": 30, "input-dim": None},
	"IMDB": {"num-classes": 2, "embedding-dim": 128, "hidden-dim": 512, "latent-dim": 30, "input-dim": None},
	"YELP": {"num-classes": 2, "embedding-dim": 128, "hidden-dim": 512, "latent-dim": 30, "input-dim": None},
}
LINEAR_RESULTS_FOLDERS = ("linear", "linear/linear")


def _resolve_project_root(project_root: Path):
	project_root = Path(project_root)
	if (project_root / "code" / "accommodation").exists():
		return project_root
	parent = project_root.parent
	if (parent / "code" / "accommodation").exists():
		return parent
	return project_root


def _load_json(path: Path):
	with open(path, "r", encoding="utf-8") as file:
		return json.load(file)


def _series_matrix(rows):
	if not rows:
		return np.empty((0, 0), dtype=float)
	max_len = max(len(row) for row in rows)
	matrix = np.full((len(rows), max_len), np.nan, dtype=float)
	for idx, row in enumerate(rows):
		arr = np.asarray(row, dtype=float)
		matrix[idx, : len(arr)] = arr
	return matrix


def _support_matrix(rows):
	processed = []
	for row in rows:
		series = []
		for item in row:
			if isinstance(item, dict):
				series.append(item.get("kappa_total", np.nan))
			elif item is None:
				series.append(np.nan)
			else:
				series.append(item)
		processed.append(series)
	return _series_matrix(processed)


def _flatten_pairs(x_matrix, y_matrix):
	if x_matrix.size == 0 or y_matrix.size == 0:
		return np.array([]), np.array([])
	length = min(x_matrix.shape[1], y_matrix.shape[1])
	x = x_matrix[:, :length].reshape(-1)
	y = y_matrix[:, :length].reshape(-1)
	mask = np.isfinite(x) & np.isfinite(y)
	return x[mask], y[mask]


def _compact_run_config(config_name: str):
	parts = {"plasticity": "False", "gamma": "0", "lambda": "0", "ppc": "5", "bbe": None}
	for part in config_name.split("_"):
		if part.startswith("PL"):
			parts["plasticity"] = part[2:]
		elif part.startswith("GM"):
			parts["gamma"] = part[2:]
		elif part.startswith("DF"):
			parts["lambda"] = part[2:]
		elif part.startswith("PPC"):
			parts["ppc"] = part[3:]
		elif part.startswith("BBE"):
			parts["bbe"] = part[3:]
	gamma = parts["gamma"] if parts["plasticity"] == "True" else "0"
	label = f"PL={parts['plasticity']}, gamma={gamma}, lambda={parts['lambda']}, PPC={parts['ppc']}"
	if parts["bbe"] is not None:
		label += f", BBE={parts['bbe']}"
	return label


def _calculate_non_neutral_plasticity(potents):
	values = [
		float(potent["plasticity"]) if potent["plasticity"] is not None else 1.0
		for potent in potents
		if potent["type"] != "Neutral"
	]
	return float(np.mean(values)) if values else np.nan


def _plasticity_matrix(records):
	rows = []
	for record in records:
		epochs = record.get("epochs", [])
		if not epochs:
			continue
		row = [1.0]
		for epoch in epochs[1:]:
			row.append(_calculate_non_neutral_plasticity(epoch["potents"]))
		rows.append(row)
	return _series_matrix(rows)


def _plasticity_matrix_from_payload(payload):
	if not payload:
		return np.empty((0, 0))
	return _series_matrix(payload.get("plasticity", []))


def _plasticity_matrix_from_payload_or_traces(viewer, dataset, config_name, payload):
	matrix = _plasticity_matrix_from_payload(payload)
	if matrix.size != 0:
		return matrix
	return _plasticity_matrix(viewer.trace_records(dataset, config_name))


def _pca_2d(matrix):
	matrix = np.asarray(matrix, dtype=float)
	if matrix.ndim != 2 or matrix.shape[0] == 0:
		return np.empty((0, 2), dtype=float)
	matrix = matrix - matrix.mean(axis=0, keepdims=True)
	_, _, vt = np.linalg.svd(matrix, full_matrices=False)
	components = vt[:2].T
	return matrix @ components


def _plot_mean_std(ax, matrix, color="tab:blue", linewidth=2.0, label=None, fill=True):
	if matrix.size == 0:
		return
	x = np.arange(matrix.shape[1])
	mean = np.nanmean(matrix, axis=0)
	std = np.nanstd(matrix, axis=0)
	ax.plot(x, mean, color=color, linewidth=linewidth, label=label)
	if fill:
		lower = np.maximum(mean - std, 0.0)
		upper = mean + std
		ax.fill_between(x, lower, upper, color=color, alpha=0.2)


def _plot_metric_nc1_scatter(ax, x_matrix, nc1_matrix, title, xlabel, color):
	ax.set_xlabel(xlabel)
	ax.grid(True, alpha=0.3)
	xs, ys = _flatten_pairs(x_matrix, nc1_matrix)
	if xs.size == 0 or ys.size == 0:
		return
	ax.scatter(xs, ys, s=8, alpha=0.3, color=color, edgecolors="none")


def _plot_matrix(ax, matrix, title, cmap="Blues", vmin=None, vmax=None):
	if matrix is None:
		if title:
			ax.set_title(title)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_box_aspect(1)
		return None
	image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
	if title:
		ax.set_title(title)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_box_aspect(1)
	return image


def _safe_pearsonr(x, y):
	x = np.asarray(x, dtype=float).reshape(-1)
	y = np.asarray(y, dtype=float).reshape(-1)
	mask = np.isfinite(x) & np.isfinite(y)
	x = x[mask]
	y = y[mask]
	if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
		return np.nan
	return float(pearsonr(x, y)[0])


def _partial_corr_controlling_epoch(x, y):
	x = np.asarray(x, dtype=float).reshape(-1)
	y = np.asarray(y, dtype=float).reshape(-1)
	if x.size == 0 or y.size == 0:
		return np.nan
	length = min(x.size, y.size)
	x = x[:length]
	y = y[:length]
	epoch = np.arange(length, dtype=float)
	mask = np.isfinite(x) & np.isfinite(y)
	x = x[mask]
	y = y[mask]
	epoch = epoch[mask]
	if x.size < 3:
		return np.nan
	design = np.column_stack([np.ones_like(epoch), epoch])
	beta_x, *_ = np.linalg.lstsq(design, x, rcond=None)
	beta_y, *_ = np.linalg.lstsq(design, y, rcond=None)
	res_x = x - design @ beta_x
	res_y = y - design @ beta_y
	return _safe_pearsonr(res_x, res_y)


def _fisher_summary(values):
	arr = np.asarray(values, dtype=float)
	arr = arr[np.isfinite(arr)]
	if arr.size == 0:
		return {"r": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
	clipped = np.clip(arr, -0.999999, 0.999999)
	z = np.arctanh(clipped)
	mean_z = float(np.mean(z))
	if z.size > 1:
		se_z = float(np.std(z, ddof=1) / np.sqrt(z.size))
		low_z = mean_z - 1.96 * se_z
		high_z = mean_z + 1.96 * se_z
	else:
		low_z = high_z = mean_z
	return {
		"r": float(np.tanh(mean_z)),
		"ci_low": float(np.tanh(low_z)),
		"ci_high": float(np.tanh(high_z)),
		"n": int(arr.size),
	}


def _permutation_p_from_run_values(values, permutations=10000, seed=42):
	arr = np.asarray(values, dtype=float)
	arr = arr[np.isfinite(arr)]
	if arr.size == 0:
		return np.nan
	rng = np.random.default_rng(seed)
	observed = abs(float(np.mean(arr)))
	signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, arr.size))
	permuted = np.abs(np.mean(signs * arr[None, :], axis=1))
	return float((np.count_nonzero(permuted >= observed) + 1) / (permutations + 1))


def _format_p_value(p):
	if not np.isfinite(p):
		return "nan"
	if p < 0.001:
		return "< 0.001"
	return f"{p:.3f}"


def _format_r_ci(summary):
	r = summary["r"]
	ci_low = summary["ci_low"]
	ci_high = summary["ci_high"]
	if not np.isfinite(r):
		return "nan"
	return f"{r:.2f} [{ci_low:.2f}, {ci_high:.2f}]"


def _combine_p_values_fisher(p_values):
	arr = np.asarray(p_values, dtype=float)
	arr = arr[np.isfinite(arr) & (arr > 0.0) & (arr <= 1.0)]
	if arr.size == 0:
		return np.nan
	statistic = float(-2.0 * np.sum(np.log(arr)))
	return float(1.0 - chi2.cdf(statistic, 2 * arr.size))


def _collect_run_level_observational_stats(payload):
	if not payload:
		return {}
	nc1 = _series_matrix(payload.get("nc1", []))
	metric_matrices = {
		"Effective support": _support_matrix(payload.get("support", [])),
		"Effective number of potents": _series_matrix(payload.get("usage", [])),
		"Mean drift": _series_matrix(payload.get("drift", [])),
	}
	results = {name: {"r": [], "partial_r": []} for name in metric_matrices}
	if nc1.size == 0:
		return results
	for name, metric_matrix in metric_matrices.items():
		n_runs = min(metric_matrix.shape[0], nc1.shape[0]) if metric_matrix.size else 0
		for run_idx in range(n_runs):
			x = metric_matrix[run_idx]
			y = nc1[run_idx]
			results[name]["r"].append(_safe_pearsonr(x, y))
			results[name]["partial_r"].append(_partial_corr_controlling_epoch(x, y))
	return results


def _summarise_observational_rows(run_level_stats, permutations=10000, seed=42):
	rows = []
	for descriptor in ("Effective support", "Effective number of potents", "Mean drift"):
		stats = run_level_stats.get(descriptor, {})
		r_values = stats.get("r", [])
		partial_values = stats.get("partial_r", [])
		r_summary = _fisher_summary(r_values)
		partial_summary = _fisher_summary(partial_values)
		rows.append({
			"descriptor": descriptor,
			"fisher_r": r_summary["r"],
			"ci_low": r_summary["ci_low"],
			"ci_high": r_summary["ci_high"],
			"partial_r": partial_summary["r"],
			"permutation_p": _permutation_p_from_run_values(r_values, permutations=permutations, seed=seed),
			"n_runs": r_summary["n"],
			"fisher_r_text": _format_r_ci(r_summary),
			"partial_r_text": "nan" if not np.isfinite(partial_summary["r"]) else f"{partial_summary['r']:.2f}",
			"permutation_p_text": _format_p_value(_permutation_p_from_run_values(r_values, permutations=permutations, seed=seed)),
		})
	return rows


def compute_observational_fidelity_table(viewer, datasets=None, permutations=10000, seed=42):
	datasets = _require_datasets(datasets or viewer.datasets)
	by_dataset = {}
	pooled = {
		"Effective support": {"r": [], "partial_r": []},
		"Effective number of potents": {"r": [], "partial_r": []},
		"Mean drift": {"r": [], "partial_r": []},
	}
	for dataset in datasets:
		run_level = _collect_run_level_observational_stats(viewer.baseline_payload(dataset))
		by_dataset[dataset] = _summarise_observational_rows(run_level, permutations=permutations, seed=seed)
		for descriptor, stats in run_level.items():
			pooled[descriptor]["r"].extend(stats["r"])
			pooled[descriptor]["partial_r"].extend(stats["partial_r"])
	macro_average = []
	for descriptor in ("Effective support", "Effective number of potents", "Mean drift"):
		dataset_rows = [rows for rows in by_dataset.values() if rows]
		descriptor_rows = [next((row for row in rows if row["descriptor"] == descriptor), None) for rows in dataset_rows]
		descriptor_rows = [row for row in descriptor_rows if row is not None]
		r_summary = _fisher_summary([row["fisher_r"] for row in descriptor_rows])
		partial_summary = _fisher_summary([row["partial_r"] for row in descriptor_rows])
		combined_p = _combine_p_values_fisher([row["permutation_p"] for row in descriptor_rows])
		macro_average.append({
			"descriptor": descriptor,
			"fisher_r": r_summary["r"],
			"ci_low": r_summary["ci_low"],
			"ci_high": r_summary["ci_high"],
			"partial_r": partial_summary["r"],
			"permutation_p": combined_p,
			"n_runs": len(descriptor_rows),
			"fisher_r_text": _format_r_ci(r_summary),
			"partial_r_text": "nan" if not np.isfinite(partial_summary["r"]) else f"{partial_summary['r']:.2f}",
			"permutation_p_text": _format_p_value(combined_p),
		})
	return {
		"overall": macro_average,
		"overall_pooled_runs": _summarise_observational_rows(pooled, permutations=permutations, seed=seed),
		"by_dataset": by_dataset,
		"datasets": list(datasets),
	}


def _draw_observational_table(rows, title=None):
	_set_default_plot_style({"axes.titlesize": 14, "axes.labelsize": 11})
	fig_h = max(1.8, 0.62 * (len(rows) + 2))
	fig, ax = plt.subplots(figsize=(11.5, fig_h))
	ax.axis("off")
	columns = [
		"Descriptor",
		"Run-level Fisher $r$ [95% CI]",
		"Partial $r$ controlling epoch",
		"Permutation $p$",
	]
	cell_text = [
		[
			row["descriptor"],
			row["fisher_r_text"],
			row["partial_r_text"],
			row["permutation_p_text"],
		]
		for row in rows
	]
	table = ax.table(
		cellText=cell_text,
		colLabels=columns,
		cellLoc="center",
		colLoc="center",
		loc="center",
		colWidths=[0.28, 0.28, 0.26, 0.18],
	)
	table.auto_set_font_size(False)
	table.set_fontsize(13)
	table.scale(1, 1.55)
	for (row_idx, col_idx), cell in table.get_celld().items():
		cell.set_edgecolor("white")
		cell.set_linewidth(0)
		if row_idx == 0:
			cell.set_text_props(weight="bold", color="black")
			cell.set_height(cell.get_height() * 1.15)
		else:
			if col_idx == 0:
				cell.set_text_props(ha="left", color="black")
			else:
				cell.set_text_props(color="red")
	if title:
		ax.set_title(title, pad=10)
	return fig


def plot_observational_fidelity_tables(viewer, datasets=None, permutations=10000, seed=42):
	summary = compute_observational_fidelity_table(viewer, datasets=datasets, permutations=permutations, seed=seed)
	output_dir = viewer.figures_dir / "observational_fidelity_tables"
	output_dir.mkdir(parents=True, exist_ok=True)
	written = {}
	overall_fig = _draw_observational_table(summary["overall"], title="Observational fidelity summary (macro-average across datasets)")
	overall_path = output_dir / "observational_fidelity_table_overall.pdf"
	_save_and_show(overall_fig, overall_path)
	written["overall"] = overall_path
	pooled_fig = _draw_observational_table(summary["overall_pooled_runs"], title="Observational fidelity summary (pooled runs)")
	pooled_path = output_dir / "observational_fidelity_table_overall_pooled_runs.pdf"
	_save_and_show(pooled_fig, pooled_path)
	written["overall_pooled_runs"] = pooled_path
	per_dataset = {}
	for dataset, rows in summary["by_dataset"].items():
		fig = _draw_observational_table(rows, title=dataset)
		path = output_dir / f"observational_fidelity_table_{dataset}.pdf"
		_save_and_show(fig, path)
		per_dataset[dataset] = path
	written["by_dataset"] = per_dataset
	return summary, written


def _set_default_plot_style(extra=None):
	style = {
		"figure.facecolor": "white",
		"axes.facecolor": "white",
		"axes.edgecolor": "black",
		"axes.labelcolor": "black",
		"xtick.color": "black",
		"ytick.color": "black",
		"text.color": "black",
		"axes.titlesize": 13,
		"axes.labelsize": 10,
		"legend.fontsize": 9,
		"xtick.labelsize": 9,
		"ytick.labelsize": 9,
	}
	if extra:
		style.update(extra)
	plt.rcParams.update(style)


def _default_device():
	if torch.cuda.is_available():
		return "cuda"
	if torch.backends.mps.is_available():
		return "mps"
	return "cpu"


def _save_and_show(fig, path: Path):
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(path, format="pdf", bbox_inches="tight", facecolor="white")
	return fig


def _reserve_top_space(fig, top=0.9):
	try:
		fig.set_constrained_layout(False)
	except Exception:
		pass
	fig.subplots_adjust(top=top)


LINE_FIGURE_HEIGHT_SCALE = 0.8
TOP_LEGEND_BBOX_Y = 0.985
TOP_LEGEND_SUBPLOTS_TOP = 0.87


def _scaled_line_height(height):
	return height * LINE_FIGURE_HEIGHT_SCALE


def _add_shared_top_legend(fig, handles, labels, *, fontsize=13, ncol=None):
	if not handles:
		return
	fig.legend(
		handles,
		labels,
		loc="upper center",
		ncol=ncol or len(labels),
		frameon=False,
		bbox_to_anchor=(0.5, TOP_LEGEND_BBOX_Y),
		fontsize=fontsize,
	)
	_reserve_top_space(fig, top=TOP_LEGEND_SUBPLOTS_TOP)


def _add_shared_top_legend_with_spacing(fig, handles, labels, *, fontsize=13, ncol=None, bbox_y=None, subplots_top=None):
	if not handles:
		return
	fig.legend(
		handles,
		labels,
		loc="upper center",
		ncol=ncol or len(labels),
		frameon=False,
		bbox_to_anchor=(0.5, TOP_LEGEND_BBOX_Y if bbox_y is None else bbox_y),
		fontsize=fontsize,
	)
	_reserve_top_space(fig, top=TOP_LEGEND_SUBPLOTS_TOP if subplots_top is None else subplots_top)


class AllDatasetsResultViewer:
	def __init__(
		self,
		project_root,
		*,
		cache_dir="visualization_cache",
		results_dir="results",
		figures_dir="figures/all_datasets",
		data_dir="data",
		device=None,
		datasets=None,
	):
		self.project_root = _resolve_project_root(Path(project_root))
		self.cache_dir = self.project_root / cache_dir
		self.results_dir = self.project_root / results_dir
		self.figures_dir = self.project_root / figures_dir
		self.data_dir = self.project_root / data_dir
		self.device = _default_device() if device is None else device
		self.figures_dir.mkdir(parents=True, exist_ok=True)
		available = [dataset for dataset in DEFAULT_DATASETS if (self.cache_dir / dataset).exists() or any((self.results_dir / config / "accommodation" / dataset).exists() for config in {RUN_GROUPS["baseline"], *RUN_GROUPS["df"].values(), *RUN_GROUPS["plasticity"].values(), *RUN_GROUPS["combined"].values(), *RUN_GROUPS["ppc_plasticity"].values(), *RUN_GROUPS["ppc_no_plasticity"].values(), *FROZEN_RUN_GROUPS.values()})]
		self.datasets = list(datasets or available)

	def cache_payload(self, dataset, config_name, experiment_type="accommodation"):
		path = self.cache_dir / dataset / experiment_type / f"{config_name}.json"
		if not path.exists():
			return None
		try:
			return _load_json(path)
		except (OSError, json.JSONDecodeError):
			return None

	def trace_records(self, dataset, config_name, experiment_type="accommodation"):
		dataset_dir = self.results_dir / config_name / experiment_type / dataset
		if not dataset_dir.exists():
			return []
		records = []
		for path in sorted(dataset_dir.glob("*.json")):
			try:
				records.append(_load_json(path))
			except (OSError, json.JSONDecodeError):
				continue
		return records

	def baseline_payload(self, dataset):
		return self.cache_payload(dataset, RUN_GROUPS["baseline"])

	def frozen_payload(self, dataset, key):
		return self.cache_payload(dataset, FROZEN_RUN_GROUPS[key], experiment_type="frozen-backbone")


def create_all_datasets_result_viewer(project_root=None, **kwargs):
	project_root = Path.cwd() if project_root is None else Path(project_root)
	return AllDatasetsResultViewer(project_root, **kwargs)


def _require_datasets(datasets):
	if datasets:
		return datasets
	raise ValueError(
		"No datasets found. Check that project_root points to the repository root "
		"and that visualization_cache/results exist there."
	)


def _baseline_trace_records(viewer, dataset):
	return viewer.trace_records(dataset, RUN_GROUPS["baseline"])


def _accuracy_series_from_trace(trace):
	series = []
	for epoch in trace.get("epochs", [])[1:]:
		metrics = epoch.get("metrics") or {}
		series.append(metrics.get("accuracy", np.nan))
	return series


def _best_accuracy_from_trace(trace):
	series = [value for value in _accuracy_series_from_trace(trace) if np.isfinite(value)]
	return float(max(series)) if series else np.nan


def _baseline_best_accuracies(viewer, dataset):
	return [
		_best_accuracy_from_trace(trace)
		for trace in _baseline_trace_records(viewer, dataset)
		if np.isfinite(_best_accuracy_from_trace(trace))
	]


def _linear_record_paths(viewer, dataset):
	paths = []
	for folder in LINEAR_RESULTS_FOLDERS:
		root = viewer.results_dir / folder / dataset
		if not root.exists():
			continue
		paths.extend(sorted(root.glob("*/cycle_*.json")))
	return paths


def _linear_best_accuracies(viewer, dataset):
	values = []
	for path in _linear_record_paths(viewer, dataset):
		try:
			record = _load_json(path)
		except (OSError, json.JSONDecodeError):
			continue
		metrics = record.get("metrics") or {}
		if "accuracy" in metrics:
			values.append(float(metrics["accuracy"]))
	return values


def _baseline_snapshot_path(viewer, dataset, trace, epoch):
	configuration = trace.get("configuration", {})
	field = configuration.get("field", {})
	policy = configuration.get("computation", {}).get("policy", "Likelihood")
	return (
		viewer.results_dir
		/ RUN_GROUPS["baseline"]
		/ "accommodation"
		/ dataset
		/ str(policy)
		/ f"POS{field.get('positive_potents')}-NEU{field.get('neutral_potents')}-NEG{field.get('negative_potents')}"
		/ str(trace["seed"])
		/ f"snapshot_{epoch:02d}.pt"
	)


def _build_val_loader(viewer, dataset):
	if dataset == "MNIST":
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
		val_dataset = MNISTDataset(str(viewer.data_dir), train=False, transform=transform)
		return DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=False), None
	if dataset == "FMNIST":
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
		val_dataset = FashionMNISTDataset(str(viewer.data_dir), train=False, transform=transform)
		return DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True, drop_last=False), None
	if dataset == "CIFAR10":
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
		])
		val_dataset = Cifar10Dataset(str(viewer.data_dir), train=False, transform=transform)
		return DataLoader(val_dataset, batch_size=256, shuffle=False, pin_memory=True, drop_last=False), None
	if dataset == "AGNEWS":
		full_dataset = AGNewsDataset(str(viewer.data_dir))
		train_size = int(0.8 * len(full_dataset))
		val_size = len(full_dataset) - train_size
		_, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
		return DataLoader(val_dataset, batch_size=64, shuffle=False), full_dataset.vocab_size
	if dataset == "IMDB":
		full_dataset = IMDBDataset(str(viewer.data_dir))
		train_size = int(0.8 * len(full_dataset))
		val_size = len(full_dataset) - train_size
		_, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
		return DataLoader(val_dataset, batch_size=64, shuffle=False), full_dataset.vocab_size
	if dataset == "YELP":
		full_dataset = YelpReviewsStarsDataset(str(viewer.data_dir))
		train_size = int(0.8 * len(full_dataset))
		val_size = len(full_dataset) - train_size
		_, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
		return DataLoader(val_dataset, batch_size=64, shuffle=False), full_dataset.vocab_size
	raise KeyError(f"Unsupported dataset {dataset}")


def _build_baseline_model(dataset, args, vocab_size=None):
	if dataset in {"MNIST", "FMNIST"}:
		return instantiate_mnist_accommodation_classifier(args)
	if dataset == "CIFAR10":
		return instantiate_cifar_accommodation_classifier(args)
	if dataset in {"AGNEWS", "IMDB", "YELP"}:
		return instantiate_gru_accommodation_classifier(args, vocab_size)
	raise KeyError(f"Unsupported dataset {dataset}")


def _baseline_model_args(dataset, trace):
	field = trace.get("configuration", {}).get("field", {})
	num_classes = trace.get("num_classes", DATASET_SPECS[dataset]["num-classes"])
	ppc = field.get("positive_potents", num_classes * 5) // max(1, num_classes)
	return {
		"dataset": dataset,
		"embedding-dim": field.get("embedding_dim", DATASET_SPECS[dataset]["embedding-dim"]),
		"hidden-dim": DATASET_SPECS[dataset]["hidden-dim"],
		"latent-dim": field.get("latent_dim", DATASET_SPECS[dataset]["latent-dim"]),
		"neutral-potents": field.get("neutral_potents", 0) // max(1, ppc),
		"num-potents-per-class": ppc,
		"negative-potents": field.get("negative_potents", 0) > 0,
		"num-classes": num_classes,
		"input-dim": DATASET_SPECS[dataset]["input-dim"],
		"plasticity": False,
	}


def _load_baseline_final_model(viewer, dataset, trace, val_vocab_size=None):
	epoch = len(trace.get("epochs", [])) - 1
	state_path = _baseline_snapshot_path(viewer, dataset, trace, epoch)
	state_dict = torch.load(state_path, map_location=viewer.device)
	args = _baseline_model_args(dataset, trace)
	model = _build_baseline_model(dataset, args, vocab_size=val_vocab_size).to(viewer.device)
	model.load_state_dict(state_dict)
	model.eval()
	return model


@torch.no_grad()
def _evaluate_model_accuracy(model, val_loader, dataset, device):
	model.eval()
	correct = 0
	total = 0
	for xb, yb in val_loader:
		xb = xb.to(device)
		yb = yb.to(device)
		if dataset in {"AGNEWS", "IMDB", "YELP"}:
			attention_mask = (xb != 0).long()
			output = model(xb, attention_mask)
		else:
			output = model(xb)
		logits = output[0] if isinstance(output, tuple) else output
		preds = torch.argmax(logits, dim=1)
		correct += (preds == yb).sum().item()
		total += yb.size(0)
	return correct / max(1, total)


def _with_perturbed_field(model, strength, noise_seed):
	layer = model.accommodation_layer
	names = (
		"positive_mu",
		"positive_sigma",
		"negative_mu",
		"negative_sigma",
		"neutral_mu",
		"neutral_sigma",
	)
	backups = {name: getattr(layer, name).detach().clone() for name in names if getattr(layer, name, None) is not None}
	with torch.no_grad():
		torch.manual_seed(noise_seed)
		for name, tensor in backups.items():
			getattr(layer, name).copy_(tensor + strength * torch.randn_like(tensor))
	return backups


def _restore_field(model, backups):
	layer = model.accommodation_layer
	with torch.no_grad():
		for name, tensor in backups.items():
			getattr(layer, name).copy_(tensor)


def plot_figure_1_linear_baseline_sensitivity(viewer, datasets=None, perturbation_strengths=(0.0, 1.0, 2.0, 3.0, 4.0)):
	datasets = _require_datasets(datasets or viewer.datasets)
	_set_default_plot_style({"legend.fontsize": 11, "axes.titlesize": 15, "axes.labelsize": 12})
	fig, axes = plt.subplots(1, 3, figsize=(20, _scaled_line_height(6)), constrained_layout=True)
	learning_ax, comparison_ax, sensitivity_ax = axes
	dataset_handles = []
	linear_means, linear_stds, baseline_means, baseline_stds = [], [], [], []
	learning_bounds = []
	sensitivity_bounds = []

	for dataset in datasets:
		color = DATASET_COLORS.get(dataset, None)
		traces = _baseline_trace_records(viewer, dataset)

		# Learning-curve accuracy comes directly from the saved per-epoch traces.
		accuracy_rows = [_accuracy_series_from_trace(trace) for trace in traces]
		accuracy_matrix = _series_matrix(accuracy_rows)
		if accuracy_matrix.size:
			x = np.arange(1, accuracy_matrix.shape[1] + 1)
			mean = np.nanmean(accuracy_matrix, axis=0)
			std = np.nanstd(accuracy_matrix, axis=0)
			handle = learning_ax.plot(x, mean, color=color, linewidth=2.4, label=dataset)[0]
			learning_ax.fill_between(x, np.maximum(mean - std, 0.0), np.minimum(mean + std, 1.0), color=color, alpha=0.18)
			dataset_handles.append(handle)
			learning_bounds.append((float(np.nanmin(np.maximum(mean - std, 0.0))), float(np.nanmax(np.minimum(mean + std, 1.0)))))

		# Best accuracies are read from serialized traces / linear json records.
		baseline_values = _baseline_best_accuracies(viewer, dataset)
		linear_values = [value for value in _linear_best_accuracies(viewer, dataset) if np.isfinite(value)]
		baseline_means.append(np.mean(baseline_values) if baseline_values else np.nan)
		baseline_stds.append(np.std(baseline_values) if baseline_values else np.nan)
		linear_means.append(np.mean(linear_values) if linear_values else np.nan)
		linear_stds.append(np.std(linear_values) if linear_values else np.nan)

		val_loader, vocab_size = _build_val_loader(viewer, dataset)
		sensitivity_values = []
		for trace in traces:
			try:
				model = _load_baseline_final_model(viewer, dataset, trace, val_vocab_size=vocab_size)
			except (FileNotFoundError, RuntimeError):
				continue
			seed_values = []
			for strength in perturbation_strengths:
				backups = _with_perturbed_field(model, float(strength), noise_seed=int(trace["seed"] * 1000 + strength * 100))
				try:
					seed_values.append(_evaluate_model_accuracy(model, val_loader, dataset, viewer.device))
				finally:
					_restore_field(model, backups)
			sensitivity_values.append(seed_values)
		sensitivity_matrix = _series_matrix(sensitivity_values)
		if sensitivity_matrix.size:
			mean = np.nanmean(sensitivity_matrix, axis=0)
			std = np.nanstd(sensitivity_matrix, axis=0)
			x = np.asarray(perturbation_strengths, dtype=float)
			sensitivity_ax.plot(x, mean, color=color, linewidth=2.4, label=dataset)
			sensitivity_ax.fill_between(x, np.maximum(mean - std, 0.0), np.minimum(mean + std, 1.0), color=color, alpha=0.18)
			sensitivity_bounds.append((float(np.nanmin(np.maximum(mean - std, 0.0))), float(np.nanmax(np.minimum(mean + std, 1.0)))))

	learning_ax.set_title("Learning Curve")
	learning_ax.set_xlabel("Epoch")
	learning_ax.set_ylabel("Accuracy")
	learning_x = np.arange(1, 81, dtype=float)
	learning_pad = 0.05 * (learning_x[-1] - learning_x[0])
	learning_ax.set_xlim(learning_x[0] - learning_pad, learning_x[-1] + learning_pad)
	if learning_bounds:
		learning_min = min(bound[0] for bound in learning_bounds)
		learning_max = max(bound[1] for bound in learning_bounds)
		learning_ax.set_ylim(max(0.0, learning_min - 0.04), min(1.02, learning_max + 0.04))
	learning_ax.grid(True, alpha=0.25)

	xpos = np.arange(len(datasets))
	point_offset = 0.14
	for idx, (linear_mean, baseline_mean) in enumerate(zip(linear_means, baseline_means)):
		if not (np.isfinite(linear_mean) and np.isfinite(baseline_mean)):
			continue
		y0 = min(linear_mean, baseline_mean)
		y1 = max(linear_mean, baseline_mean)
		color = "tab:green" if baseline_mean >= linear_mean else "tab:red"
		comparison_ax.fill_between(
			[idx - 0.24, idx + 0.24],
			[y0, y0],
			[y1, y1],
			color=color,
			alpha=0.18,
			zorder=1,
		)
	comparison_ax.errorbar(xpos - point_offset, linear_means, yerr=linear_stds, fmt="o", color="tab:blue", capsize=5, markersize=11, label="Linear Head")
	comparison_ax.errorbar(xpos + point_offset, baseline_means, yerr=baseline_stds, fmt="o", color="tab:orange", capsize=5, markersize=11, label="Accommodation Head")
	comparison_ax.set_title("Model Comparison")
	comparison_ax.set_xticks(xpos, datasets, rotation=40, ha="right")
	comparison_values = []
	for mean, std in zip(linear_means, linear_stds):
		if np.isfinite(mean):
			comparison_values.append((mean - (std if np.isfinite(std) else 0.0), mean + (std if np.isfinite(std) else 0.0)))
	for mean, std in zip(baseline_means, baseline_stds):
		if np.isfinite(mean):
			comparison_values.append((mean - (std if np.isfinite(std) else 0.0), mean + (std if np.isfinite(std) else 0.0)))
	if comparison_values:
		comp_min = min(value[0] for value in comparison_values)
		comp_max = max(value[1] for value in comparison_values)
		comparison_ax.set_ylim(max(0.0, comp_min - 0.04), min(1.02, comp_max + 0.04))
	comparison_ax.grid(True, axis="y", alpha=0.25)
	comparison_ax.legend(loc="upper right", frameon=True)

	sensitivity_ax.set_title("Sensitivity")
	sensitivity_ax.set_xlabel("Perturbation Strength")
	if sensitivity_bounds:
		sens_min = min(bound[0] for bound in sensitivity_bounds)
		sens_max = max(bound[1] for bound in sensitivity_bounds)
		sensitivity_ax.set_ylim(max(0.0, sens_min - 0.04), min(1.02, sens_max + 0.04))
	sensitivity_ax.grid(True, alpha=0.25)

	if dataset_handles:
		_add_shared_top_legend_with_spacing(
			fig,
			dataset_handles,
			[handle.get_label() for handle in dataset_handles],
			fontsize=17,
			ncol=len(dataset_handles),
			bbox_y=1.1,
			subplots_top=0.83,
		)
	return _save_and_show(fig, viewer.figures_dir / "figure_1_linear_baseline_sensitivity_all_datasets.pdf")


def plot_figure_2_baseline(viewer, datasets=None):
	datasets = _require_datasets(datasets or viewer.datasets)
	_set_default_plot_style()
	metrics = (
		("Drift", "drift", _series_matrix),
		("Effective Support", "support", _support_matrix),
		("Effective number of potents", "usage", _series_matrix),
	)
	fig, axes = plt.subplots(len(metrics), len(datasets), figsize=(max(3 * len(datasets), 6), _scaled_line_height(9)), squeeze=False, constrained_layout=True)
	for col, dataset in enumerate(datasets):
		payload = viewer.baseline_payload(dataset)
		axes[0, col].set_title(dataset)
		for row, (title, key, matrix_fn) in enumerate(metrics):
			ax = axes[row, col]
			matrix = matrix_fn(payload[key]) if payload else np.empty((0, 0))
			_plot_mean_std(ax, matrix)
			ax.grid(True, alpha=0.3)
			if row == len(metrics) - 1:
				ax.set_xlabel("Epoch")
			if col == 0:
				ax.set_ylabel(title)
	return _save_and_show(fig, viewer.figures_dir / "figure_2_all_datasets.pdf")


def _plot_config_comparison_grid(viewer, run_lookup, labels, colors, save_name, title, include_plasticity=False, datasets=None):
	datasets = _require_datasets(datasets or viewer.datasets)
	nrows = 4 if include_plasticity else 3
	_set_default_plot_style()
	fig, axes = plt.subplots(nrows, len(datasets), figsize=(max(3 * len(datasets), 6), _scaled_line_height(3 * nrows)), squeeze=False, constrained_layout=True)
	legend_handles = []
	legend_labels = []
	for col, dataset in enumerate(datasets):
		axes[0, col].set_title(dataset)
		payloads = {label: viewer.cache_payload(dataset, run_lookup[label]) for label in labels}
		metric_specs = [
			("Drift", "drift", _series_matrix),
			("Effective Support", "support", _support_matrix),
			("Effective number of potents", "usage", _series_matrix),
		]
		if include_plasticity:
			metric_specs.append(("Plasticity", "plasticity", None))
		for row, (metric_title, key, matrix_fn) in enumerate(metric_specs):
			ax = axes[row, col]
			for label in labels:
				if key == "plasticity":
					matrix = _plasticity_matrix_from_payload_or_traces(
						viewer,
						dataset,
						run_lookup[label],
						payloads[label],
					)
				else:
					payload = payloads[label]
					matrix = matrix_fn(payload[key]) if payload else np.empty((0, 0))
				_plot_mean_std(
					ax,
					matrix,
					color=colors[label],
					label=label,
					linewidth=2.2 if label == "Baseline" else 1.6,
					fill=label != "Baseline",
				)
				if row == 0 and col == 0:
					handle = ax.lines[-1] if ax.lines else None
					if handle is not None:
						legend_handles.append(handle)
						legend_labels.append(label)
			ax.grid(True, alpha=0.3)
			if row == len(metric_specs) - 1:
				ax.set_xlabel("Epoch")
			if col == 0:
				ax.set_ylabel(metric_title)
	if legend_handles:
		_add_shared_top_legend(fig, legend_handles, legend_labels, fontsize=13, ncol=len(legend_labels))
	return _save_and_show(fig, viewer.figures_dir / save_name)


def plot_figure_3_differentiation_strength(viewer, datasets=None):
	run_lookup = {
		"$\\lambda=0.5$": RUN_GROUPS["df"]["DF-0.5"],
		"$\\lambda=0.8$": RUN_GROUPS["df"]["DF-0.8"],
		"$\\lambda=0.9$": RUN_GROUPS["df"]["DF-0.9"],
		"$\\lambda=0.95$": RUN_GROUPS["df"]["DF-0.95"],
	}
	run_lookup["Baseline"] = RUN_GROUPS["baseline"]
	colors = {
		"$\\lambda=0.5$": "tab:blue",
		"$\\lambda=0.8$": "tab:orange",
		"$\\lambda=0.9$": "tab:green",
		"$\\lambda=0.95$": "tab:red",
		"Baseline": "black",
	}
	return _plot_config_comparison_grid(
		viewer,
		run_lookup,
		["$\\lambda=0.5$", "$\\lambda=0.8$", "$\\lambda=0.9$", "$\\lambda=0.95$", "Baseline"],
		colors,
		"figure_3_all_datasets.pdf",
		"Figure 3: Effect of differentiation strength across datasets",
		datasets=datasets,
	)


def plot_figure_4_plasticity_strength(viewer, datasets=None):
	run_lookup = {
		"$\\gamma=1$": RUN_GROUPS["plasticity"]["PL-1"],
		"$\\gamma=5$": RUN_GROUPS["plasticity"]["PL-5"],
		"$\\gamma=10$": RUN_GROUPS["plasticity"]["PL-10"],
		"$\\gamma=20$": RUN_GROUPS["plasticity"]["PL-20"],
	}
	run_lookup["Baseline"] = RUN_GROUPS["baseline"]
	colors = {
		"$\\gamma=1$": "tab:blue",
		"$\\gamma=5$": "tab:orange",
		"$\\gamma=10$": "tab:green",
		"$\\gamma=20$": "tab:red",
		"Baseline": "black",
	}
	return _plot_config_comparison_grid(
		viewer,
		run_lookup,
		["$\\gamma=1$", "$\\gamma=5$", "$\\gamma=10$", "$\\gamma=20$", "Baseline"],
		colors,
		"figure_4_all_datasets.pdf",
		"Figure 4: Effect of plasticity strength across datasets",
		include_plasticity=True,
		datasets=datasets,
	)


def plot_figure_5_joint_differentiation_plasticity(viewer, datasets=None):
	run_lookup = {
		"$\\lambda=0.5$": RUN_GROUPS["combined"]["DF-0.5"],
		"$\\lambda=0.8$": RUN_GROUPS["combined"]["DF-0.8"],
		"$\\lambda=0.9$": RUN_GROUPS["combined"]["DF-0.9"],
		"$\\lambda=0.95$": RUN_GROUPS["combined"]["DF-0.95"],
	}
	run_lookup["Baseline"] = RUN_GROUPS["baseline"]
	colors = {
		"$\\lambda=0.5$": "tab:blue",
		"$\\lambda=0.8$": "tab:orange",
		"$\\lambda=0.9$": "tab:green",
		"$\\lambda=0.95$": "tab:red",
		"Baseline": "black",
	}
	return _plot_config_comparison_grid(
		viewer,
		run_lookup,
		["$\\lambda=0.5$", "$\\lambda=0.8$", "$\\lambda=0.9$", "$\\lambda=0.95$", "Baseline"],
		colors,
		"figure_5_all_datasets.pdf",
		"Figure 5: Joint effect of differentiation and plasticity across datasets",
		datasets=datasets,
	)


def plot_figure_observational_fidelity(viewer, datasets=None):
	datasets = _require_datasets(datasets or viewer.datasets)
	_set_default_plot_style()
	row_specs = (
		("NC1", None),
		("NC1", "support"),
		("NC1", "usage"),
		("NC1", "drift"),
	)
	fig, axes = plt.subplots(len(row_specs), len(datasets), figsize=(max(3 * len(datasets), 6), _scaled_line_height(12)), squeeze=False, constrained_layout=True)
	for col, dataset in enumerate(datasets):
		payload = viewer.baseline_payload(dataset)
		axes[0, col].set_title(dataset)
		nc1 = _series_matrix(payload["nc1"]) if payload else np.empty((0, 0))
		support = _support_matrix(payload["support"]) if payload else np.empty((0, 0))
		usage = _series_matrix(payload["usage"]) if payload else np.empty((0, 0))
		drift = _series_matrix(payload["drift"]) if payload else np.empty((0, 0))
		_plot_mean_std(axes[0, col], nc1, color="tab:orange")
		axes[0, col].set_xlabel("Epoch")
		axes[0, col].grid(True, alpha=0.3)
		_plot_metric_nc1_scatter(axes[1, col], support, nc1, "Effective Support vs NC1", "Effective Support", "tab:blue")
		_plot_metric_nc1_scatter(axes[2, col], usage, nc1, "Effective Potents vs NC1", "Effective number of potents", "tab:green")
		_plot_metric_nc1_scatter(axes[3, col], drift, nc1, "Drift vs NC1", "Drift", "tab:red")
	for row, (ylabel, _) in enumerate(row_specs):
		axes[row, 0].set_ylabel(ylabel)
	return _save_and_show(fig, viewer.figures_dir / "observational_fidelity_all_datasets.pdf")


def plot_frozen_backbone_latent_dynamics(viewer, datasets=None):
	datasets = _require_datasets(datasets or viewer.datasets)
	_set_default_plot_style()
	titles = ("Drift", "Effective Support", "NC1", "Temporal potent CKA")
	fig, axes = plt.subplots(len(titles), len(datasets), figsize=(max(3 * len(datasets), 6), _scaled_line_height(12)), squeeze=False, constrained_layout=True)
	legend_handles = []
	legend_labels = []
	for col, dataset in enumerate(datasets):
		axes[0, col].set_title(dataset)
		run_specs = [
			("Baseline", viewer.frozen_payload(dataset, "baseline"), "tab:blue"),
			("$\\lambda=0.8, \\gamma=5$", viewer.frozen_payload(dataset, "combined"), "tab:orange"),
		]
		for label, payload, color in run_specs:
			drift = _series_matrix(payload["drift"]) if payload else np.empty((0, 0))
			support = _support_matrix(payload["support"]) if payload else np.empty((0, 0))
			nc1 = _series_matrix(payload["nc1"]) if payload else np.empty((0, 0))
			cka = _series_matrix(payload["cka"]) if payload else np.empty((0, 0))
			for row, matrix in enumerate((drift, support, nc1, cka)):
				_plot_mean_std(axes[row, col], matrix, color=color, label=label, fill=label != "Baseline")
				if row == 0 and col == 0 and axes[row, col].lines:
					legend_handles.append(axes[row, col].lines[-1])
					legend_labels.append(label)
		for row, title in enumerate(titles):
			ax = axes[row, col]
			if row == len(titles) - 1:
				ax.set_xlabel("Epoch")
			else:
				ax.set_xlabel("")
			ax.grid(True, alpha=0.3)
			if col == 0:
				ax.set_ylabel(title)
		axes[2, col].set_ylim(0.0, 1.0)
		axes[3, col].set_ylim(0.0, 1.05)
	if legend_handles:
		_add_shared_top_legend(fig, legend_handles, legend_labels, fontsize=13, ncol=len(legend_labels))
	return _save_and_show(fig, viewer.figures_dir / "frozen_backbone_latent_dynamics_all_datasets.pdf")


def plot_figure_6_potents_per_class(viewer, datasets=None):
	datasets = _require_datasets(datasets or viewer.datasets)
	_set_default_plot_style()
	labels = ["PPC-1", "PPC-5", "PPC-10", "PPC-20"]
	colors = {"PPC-1": "tab:blue", "PPC-5": "tab:orange", "PPC-10": "tab:green", "PPC-20": "tab:red"}
	row_specs = (
		("ppc_no_plasticity", "$\\lambda=0, \\gamma=0$", "Drift", "drift", _series_matrix),
		("ppc_no_plasticity", "$\\lambda=0, \\gamma=0$", "Effective Support", "support", _support_matrix),
		("ppc_no_plasticity", "$\\lambda=0, \\gamma=0$", "Effective number of potents", "usage", _series_matrix),
		("ppc_plasticity", "$\\lambda=0.8, \\gamma=5$", "Drift", "drift", _series_matrix),
		("ppc_plasticity", "$\\lambda=0.8, \\gamma=5$", "Effective Support", "support", _support_matrix),
		("ppc_plasticity", "$\\lambda=0.8, \\gamma=5$", "Effective number of potents", "usage", _series_matrix),
	)
	fig, axes = plt.subplots(len(row_specs), len(datasets), figsize=(max(3 * len(datasets), 6), _scaled_line_height(16)), squeeze=False, constrained_layout=True)
	legend_handles = []
	legend_labels = []
	for col, dataset in enumerate(datasets):
		axes[0, col].set_title(dataset)
		for row, (group_name, subtitle, metric_title, key, matrix_fn) in enumerate(row_specs):
			for label in labels:
				payload = viewer.cache_payload(dataset, RUN_GROUPS[group_name][label])
				matrix = matrix_fn(payload[key]) if payload else np.empty((0, 0))
				_plot_mean_std(axes[row, col], matrix, color=colors[label], label=label)
				if row == 0 and col == 0 and axes[row, col].lines:
					legend_handles.append(axes[row, col].lines[-1])
					legend_labels.append(label)
			if row == len(row_specs) - 1:
				axes[row, col].set_xlabel("Epoch")
			else:
				axes[row, col].set_xlabel("")
			axes[row, col].grid(True, alpha=0.3)
			if col == 0:
				axes[row, col].set_ylabel(f"{subtitle}\n{metric_title}")
	if legend_handles:
		_add_shared_top_legend_with_spacing(
			fig,
			legend_handles,
			legend_labels,
			fontsize=13,
			ncol=len(legend_labels),
			bbox_y=0.945,
			subplots_top=0.92,
		)
	return _save_and_show(fig, viewer.figures_dir / "figure_6_all_datasets.pdf")


def plot_figure_7_potent_trajectories(viewer, datasets=None):
	datasets = _require_datasets(datasets or viewer.datasets)
	_set_default_plot_style({"legend.fontsize": 8})
	color_map = {"Positive": "green", "Negative": "red"}
	run_specs = (
		("baseline", "Baseline", RUN_GROUPS["baseline"]),
		("combined", "$\\lambda=0.8, \\gamma=5$", RUN_GROUPS["combined"]["DF-0.8"]),
	)
	figures = {}
	for suffix, title, config_name in run_specs:
		fig, axes = plt.subplots(2, len(datasets), figsize=(max(3 * len(datasets), 6), _scaled_line_height(6)), squeeze=False, constrained_layout=True)
		legend_handles = []
		legend_labels = []
		for col, dataset in enumerate(datasets):
			axes[0, col].set_title(dataset, pad=2)
			payload_bundle = viewer.cache_payload(dataset, config_name)
			payload = payload_bundle.get("seed42_trajectory") if payload_bundle else None
			if not payload:
				for row in range(2):
					axes[row, col].set_xticks([])
					axes[row, col].set_yticks([])
				continue
			traj = np.array(payload["trajectory"], dtype=float)
			mov = np.array(payload["movement"], dtype=float)
			types = np.array(payload["types"])
			if traj.size == 0:
				continue
			traj_2d = _pca_2d(traj.reshape(-1, traj.shape[-1])).reshape(traj.shape[0], traj.shape[1], 2)
			for potent_idx in range(traj_2d.shape[1]):
				path = traj_2d[:, potent_idx]
				axes[0, col].plot(path[:, 0], path[:, 1], color=color_map.get(types[potent_idx], "gray"), alpha=0.6, linewidth=0.8)
			if mov.size:
				epochs = np.arange(mov.shape[0])
				for potent_type, color in color_map.items():
					mask = types == potent_type
					if np.any(mask):
						axes[1, col].plot(epochs, mov[:, mask].mean(axis=1), color=color, linewidth=1.8, label=potent_type)
						if col == 0 and potent_type not in legend_labels and axes[1, col].lines:
							legend_handles.append(axes[1, col].lines[-1])
							legend_labels.append(potent_type)
			axes[0, col].grid(True, alpha=0.3)
			axes[1, col].grid(True, alpha=0.3)
			axes[0, col].set_xticks([])
			axes[0, col].set_yticks([])
			axes[1, col].set_xlabel("Epoch")
		axes[0, 0].set_ylabel("Potent trajectories")
		axes[1, 0].set_ylabel("Per-potent drift")
		if legend_handles:
			_add_shared_top_legend_with_spacing(
				fig,
				legend_handles,
				legend_labels,
				fontsize=13,
				ncol=len(legend_labels),
				bbox_y=1.05,
				subplots_top=0.89,
			)
		figures[suffix] = _save_and_show(fig, viewer.figures_dir / f"figure_7_{suffix}_all_datasets.pdf")
	return figures


def plot_figure_8_potent_semantics(viewer, datasets=None, top_k=8, num_potents=8):
	datasets = [dataset for dataset in _require_datasets(datasets or viewer.datasets) if dataset in IMAGE_DATASETS]
	_set_default_plot_style()
	figures = {}
	for dataset in datasets:
		payload = viewer.cache_payload(dataset, RUN_GROUPS["combined"]["DF-0.8"])
		semantics = payload.get("seed42_semantics") if payload else None
		class_names = IMAGE_CLASS_NAMES.get(dataset, [])
		fig, axes = plt.subplots(num_potents, top_k, figsize=(top_k * 0.55, max(num_potents * 0.55, 4)), squeeze=False)
		if not semantics:
			for row in range(num_potents):
				for col in range(top_k):
					axes[row, col].axis("off")
			fig.subplots_adjust(left=0.06, right=0.995, top=0.94, bottom=0.02, wspace=0.005, hspace=0.005)
			figures[dataset] = _save_and_show(fig, viewer.figures_dir / f"figure_8_{dataset.lower()}_all_datasets.pdf")
			continue
		top_idx = semantics.get("top_idx", [])[:num_potents]
		meta = [tuple(item) for item in semantics.get("meta", [])]
		top_images = {int(key): value for key, value in semantics.get("top_images", {}).items()}
		for row in range(num_potents):
			if row >= len(top_idx):
				for col in range(top_k):
					axes[row, col].axis("off")
				continue
			idx = top_idx[row]
			clazz, potent_idx, potent_type = meta[idx]
			class_label = class_names[clazz] if 0 <= clazz < len(class_names) else str(clazz)
			polarity = "+" if potent_type == "Positive" else "-"
			for col in range(top_k):
				ax = axes[row, col]
				images = top_images.get(idx, [])
				if col < len(images):
					image = np.array(images[col], dtype=float)
					if image.ndim == 2:
						ax.imshow(image, cmap="gray", interpolation="nearest")
					else:
						ax.imshow(image, interpolation="nearest")
				ax.axis("off")
				if col == 0:
					ax.text(
						-0.25,
						0.5,
						f"{polarity}[{class_label}]$_{{{potent_idx}}}$",
						transform=ax.transAxes,
						ha="right",
						va="center",
						fontsize=10,
						color="black",
					)
		fig.subplots_adjust(left=0.06, right=0.995, top=0.94, bottom=0.02, wspace=0.005, hspace=0.005)
		figures[dataset] = _save_and_show(fig, viewer.figures_dir / f"figure_8_{dataset.lower()}_all_datasets.pdf")
	return figures


def plot_figure_9_temporal_potent_cka(viewer, datasets=None):
	datasets = _require_datasets(datasets or viewer.datasets)
	_set_default_plot_style()
	configs = (("Baseline", RUN_GROUPS["baseline"], "tab:blue"), ("$\\lambda=0.8, \\gamma=5$", RUN_GROUPS["combined"]["DF-0.8"], "tab:orange"))
	fig, axes = plt.subplots(len(configs), len(datasets), figsize=(max(3 * len(datasets), 6), _scaled_line_height(6)), squeeze=False, constrained_layout=True)
	for col, dataset in enumerate(datasets):
		axes[0, col].set_title(dataset)
		for row, (title, config_name, color) in enumerate(configs):
			payload = viewer.cache_payload(dataset, config_name)
			matrix = _series_matrix(payload["cka"]) if payload else np.empty((0, 0))
			_plot_mean_std(axes[row, col], matrix, color=color)
			if row == len(configs) - 1:
				axes[row, col].set_xlabel("Epoch")
			else:
				axes[row, col].set_xlabel("")
			axes[row, col].set_ylim(0.0, 1.05)
			axes[row, col].grid(True, alpha=0.3)
			if col == 0:
				axes[row, col].set_ylabel(title)
	return _save_and_show(fig, viewer.figures_dir / "figure_9_all_datasets.pdf")


def plot_differentiation_tensors_by_epoch(viewer, datasets=None, snapshots=(1, 40, 80)):
	datasets = _require_datasets(datasets or viewer.datasets)
	configs = [("Baseline", RUN_GROUPS["baseline"]), ("$\\lambda=0.5$", RUN_GROUPS["combined"]["DF-0.5"]), ("$\\lambda=0.8$", RUN_GROUPS["combined"]["DF-0.8"]), ("$\\lambda=0.95$", RUN_GROUPS["combined"]["DF-0.95"])]
	figures = {}
	for epoch in snapshots:
		_set_default_plot_style()
		fig, axes = plt.subplots(
			len(configs),
			len(datasets),
			figsize=(max(3 * len(datasets), 6), max(3 * len(configs), 4)),
			squeeze=False,
			constrained_layout=True,
		)
		vmin = None
		vmax = None
		matrices = {}
		for dataset in datasets:
			for label, config_name in configs:
				payload = viewer.cache_payload(dataset, config_name)
				matrix = None
				if payload:
					diffs = payload.get("differentiation_matrices") or []
					if 0 <= epoch < len(diffs):
						matrix = diffs[epoch]
				if matrix is not None:
					mat = np.array(matrix, dtype=float)
					matrices[(dataset, label)] = mat
					vmin = mat.min() if vmin is None else min(vmin, mat.min())
					vmax = mat.max() if vmax is None else max(vmax, mat.max())
				else:
					matrices[(dataset, label)] = None
		last_image = None
		for col, dataset in enumerate(datasets):
			for row, (label, _) in enumerate(configs):
				last_image = _plot_matrix(axes[row, col], matrices[(dataset, label)], dataset if row == 0 else "", vmin=vmin, vmax=vmax) or last_image
				if col == 0:
					axes[row, col].set_ylabel(label)
		if last_image is not None:
			fig.colorbar(last_image, ax=axes, fraction=0.02, pad=0.02)
		figures[epoch] = _save_and_show(fig, viewer.figures_dir / f"differentiation_tensors_epoch_{epoch}_all_datasets.pdf")
	return figures


def plot_final_differentiation_matrices(viewer, datasets=None):
	datasets = _require_datasets(datasets or viewer.datasets)
	configs = [
		("Baseline", RUN_GROUPS["baseline"]),
		("$\\gamma=5$", RUN_GROUPS["plasticity"]["PL-5"]),
		("$\\lambda=0.8$", RUN_GROUPS["df"]["DF-0.8"]),
		("$\\lambda=0.8, \\gamma=5$", RUN_GROUPS["combined"]["DF-0.8"]),
	]
	ppc_configs = [(label, name) for label, name in RUN_GROUPS["ppc_plasticity"].items()]
	def _build_grid(figure_title, items, file_name):
		_set_default_plot_style()
		fig, axes = plt.subplots(
			len(items),
			len(datasets),
			figsize=(max(3 * len(datasets), 6), max(3 * len(items), 4)),
			squeeze=False,
			constrained_layout=True,
		)
		vmin = None
		vmax = None
		matrices = {}
		for dataset in datasets:
			for label, config_name in items:
				payload = viewer.cache_payload(dataset, config_name)
				matrix = None if payload is None else payload.get("final_differentiation_matrix")
				if matrix is not None:
					mat = np.array(matrix, dtype=float)
					matrices[(dataset, label)] = mat
					vmin = mat.min() if vmin is None else min(vmin, mat.min())
					vmax = mat.max() if vmax is None else max(vmax, mat.max())
				else:
					matrices[(dataset, label)] = None
		last_image = None
		for col, dataset in enumerate(datasets):
			for row, (label, _) in enumerate(items):
				last_image = _plot_matrix(axes[row, col], matrices[(dataset, label)], dataset if row == 0 else "", vmin=vmin, vmax=vmax) or last_image
				if col == 0:
					axes[row, col].set_ylabel(label)
		if last_image is not None:
			fig.colorbar(last_image, ax=axes, fraction=0.02, pad=0.02)
		return _save_and_show(fig, viewer.figures_dir / file_name)
	return {
		"configuration_matrices": _build_grid("Final differentiation matrices across datasets", configs, "final_differentiation_matrices_all_datasets.pdf"),
		"ppc_matrices": _build_grid("Final differentiation matrices across datasets (PPC)", ppc_configs, "final_differentiation_ppc_all_datasets.pdf"),
	}
