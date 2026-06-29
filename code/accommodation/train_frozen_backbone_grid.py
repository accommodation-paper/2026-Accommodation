import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import torch

from accommodation.experiments.helpers.results_accessor import result_path
from accommodation.main import run_frozen_backbone_experiment
from accommodation.dataset_visualization_builder import build_visualization_cache


DEFAULTS = {
	"num-cycles": 10,
	"type": "frozen-backbone",
	"device": "cuda",
	"auto-device": False,
	"base-seed": 42,
	"data-path": "data",
	"epochs": 40,
	"backbone-epochs": 40,
	"embedding-dim": 128,
	"hidden-dim": 512,
	"num-potents-per-class": 5,
	"neutral-potents": 10,
	"negative-potents": True,
	"latent-dim": 30,
	"plasticity": True,
	"plasticity-gamma": 5,
	"differentiation-lambda": 0.5,
	"results-dir": "results",
	"parallel-cycles": True,
	"cycle-workers": 10,
	"dry-run": False,
}


DATASET_SPECS = {
	"MNIST": {"num-classes": 10},
	"FMNIST": {"num-classes": 10},
	"CIFAR10": {"num-classes": 10, "embedding-dim": 256},
	"AGNEWS": {"num-classes": 4},
	"IMDB": {"num-classes": 2},
	"YELP": {"num-classes": 2},
}


RULES = [
	{"plasticity": [False], "differentiation-lambda": [0], "num-potents-per-class": [5]},
	{"plasticity": [True], "plasticity-gamma": [5], "differentiation-lambda": [0.8], "num-potents-per-class": [5]},
]


def default_device():
	if torch.cuda.is_available():
		return "cuda"
	if torch.backends.mps.is_available():
		return "mps"
	return "cpu"


def create_settings(**overrides):
	settings = {**DEFAULTS, **overrides}
	settings.setdefault("datasets", list(DATASET_SPECS.keys()))
	settings.setdefault("rules", RULES)
	return SimpleNamespace(**settings)


def expand(rule):
	keys = list(rule.keys())
	values = list(rule.values())
	for combo in product(*values):
		yield dict(zip(keys, combo))


def cycle_seed(config, cycle_index):
	return config["base-seed"] + cycle_index


def cycle_candidates(cycle_index):
	if cycle_index == 0:
		return [0]
	return [0, cycle_index]


def summary_paths(config, cycle_index, seed):
	return [Path(result_path(config["type"], config, seed, candidate)) for candidate in cycle_candidates(cycle_index)]


def snapshot_dir(config, seed):
	pospot = config["num-classes"] * config["num-potents-per-class"]
	neutral = config["neutral-potents"] * config["num-potents-per-class"]
	negpot = pospot if config["negative-potents"] else 0
	return (
		Path(config["results-dir"])
		/ config["type"]
		/ config["dataset"]
		/ "Likelihood"
		/ f"POS{pospot}-NEU{neutral}-NEG{negpot}"
		/ str(seed)
	)


def backbone_paths(config, cycle_index, seed):
	base = Path(config["results-dir"]) / config["type"] / config["dataset"] / "pretrained_backbones" / str(seed)
	return [
		(base / f"cycle_{candidate:02d}.json", base / f"cycle_{candidate:02d}.pt", base)
		for candidate in cycle_candidates(cycle_index)
	]


def valid_serialized_trace(summary_path, expected_epochs):
	try:
		with open(summary_path, "r", encoding="utf-8") as file:
			record = json.load(file)
	except (OSError, json.JSONDecodeError):
		return None

	epochs = record.get("epochs")
	if not isinstance(epochs, list) or len(epochs) != expected_epochs + 1:
		return None
	if any(epoch.get("epoch") != idx for idx, epoch in enumerate(epochs)):
		return None
	if not all(epoch.get("metrics") is not None for epoch in epochs[1:]):
		return None
	return record


def inspect_cycle(config, cycle_index):
	seed = cycle_seed(config, cycle_index)

	valid_summary = any(
		path.exists() and valid_serialized_trace(path, config["epochs"]) is not None
		for path in summary_paths(config, cycle_index, seed)
	)
	if not valid_summary:
		return False, "missing or invalid serialized trace"

	trace = None
	for path in summary_paths(config, cycle_index, seed):
		if not path.exists():
			continue
		trace = valid_serialized_trace(path, config["epochs"])
		if trace is not None:
			break
	if trace and trace.get("visualization") is not None:
		for meta_path, state_path, _ in backbone_paths(config, cycle_index, seed):
			if meta_path.exists() and state_path.exists():
				return True, None

	snapshots = sorted(snapshot_dir(config, seed).glob("snapshot_*.pt"))
	if len(snapshots) not in {1, config["epochs"]}:
		return False, f"expected {config['epochs']} snapshots, found {len(snapshots)}"

	for meta_path, state_path, _ in backbone_paths(config, cycle_index, seed):
		if meta_path.exists() and state_path.exists():
			return True, None
	return False, "missing frozen-backbone pretraining artifacts"


def cleanup_cycle(config, cycle_index):
	seed = cycle_seed(config, cycle_index)

	for path in summary_paths(config, cycle_index, seed):
		if path.exists():
			path.unlink()

	shutil.rmtree(snapshot_dir(config, seed), ignore_errors=True)

	for _, _, base_dir in backbone_paths(config, cycle_index, seed):
		if base_dir.exists():
			shutil.rmtree(base_dir, ignore_errors=True)


def config_name(config):
	return (
		f"PL{config.get('plasticity')}"
		f"_GM{config.get('plasticity-gamma')}"
		f"_DF{config.get('differentiation-lambda')}"
		f"_PPC{config.get('num-potents-per-class')}"
		f"_BBE{config.get('backbone-epochs')}"
	)


def build_base_config(settings, dataset_name):
	spec = DATASET_SPECS[dataset_name]
	return {
		**DEFAULTS,
		"dataset": dataset_name,
		"type": "frozen-backbone",
		"device": default_device() if getattr(settings, "auto-device", False) else settings.device,
		"data-path": settings.__dict__["data-path"],
		"results-dir": settings.__dict__["results-dir"],
		"epochs": settings.epochs,
		"backbone-epochs": settings.__dict__["backbone-epochs"],
		"num-cycles": settings.__dict__["num-cycles"],
		"base-seed": settings.__dict__["base-seed"],
		"embedding-dim": settings.__dict__["embedding-dim"],
		"hidden-dim": settings.__dict__["hidden-dim"],
		"latent-dim": settings.__dict__["latent-dim"],
		"neutral-potents": settings.__dict__["neutral-potents"],
		"negative-potents": settings.__dict__["negative-potents"],
		**spec,
	}


def iter_configs(settings):
	for dataset_name in settings.datasets:
		base_config = build_base_config(settings, dataset_name)
		for rule in settings.rules:
			for combo in expand(rule):
				config = {**base_config, **combo}
				config["results-dir"] = os.path.join(settings.__dict__["results-dir"], config_name(config))
				yield config


def print_config(exp_id, config):
	print("\n" + "=" * 90, flush=True)
	print(f"EXPERIMENT {exp_id}", flush=True)
	print(config["dataset"], flush=True)
	print(
		f"plasticity={config.get('plasticity')} | "
		f"gamma={config.get('plasticity-gamma') if config.get('plasticity') else 0} | "
		f"diff={config.get('differentiation-lambda')} | "
		f"potents={config.get('num-potents-per-class')} | "
		f"backbone_epochs={config.get('backbone-epochs')} | "
		f"cycles={config.get('num-cycles')}",
		flush=True,
	)
	print(f"results={config['results-dir']}", flush=True)
	print("=" * 90 + "\n", flush=True)


def print_cycle_plan(config):
	for cycle_index in range(config["num-cycles"]):
		seed = cycle_seed(config, cycle_index)
		complete, reason = inspect_cycle(config, cycle_index)
		status = "done" if complete else f"pending ({reason})"
		print(f"cycle={cycle_index + 1:02d} seed={seed} status={status}", flush=True)


def run_cycle(config, cycle_index):
	cleanup_cycle(config, cycle_index)
	cycle_config = {
		**config,
		"num-cycles": 1,
		"base-seed": cycle_seed(config, cycle_index),
	}
	run_frozen_backbone_experiment(cycle_config)
	return cycle_index, cycle_config["base-seed"]


def pending_cycles(config):
	cycles = []
	for cycle_index in range(config["num-cycles"]):
		complete, reason = inspect_cycle(config, cycle_index)
		if complete:
			continue
		cycles.append((cycle_index, reason))
	return cycles


def run_config_cycles(config, max_workers):
	cycles = pending_cycles(config)
	if not cycles:
		print("All cycles already complete, skipping.", flush=True)
		return
	for cycle_index, reason in cycles:
		print(f"Cycle {cycle_index + 1:02d} scheduled | seed={cycle_seed(config, cycle_index)} | {reason}", flush=True)

	workers = min(max_workers, len(cycles))
	with ProcessPoolExecutor(max_workers=workers) as executor:
		futures = [
			executor.submit(run_cycle, config, cycle_index)
			for cycle_index, _ in cycles
		]
		for future in as_completed(futures):
			cycle_index, seed = future.result()
			print(f"Cycle {cycle_index + 1:02d} finished | seed={seed}", flush=True)


def maybe_build_dataset_visualization_cache(dataset_name, settings):
	project_root = Path(__file__).resolve().parents[1]
	written = build_visualization_cache(
		results_dir=settings.__dict__["results-dir"],
		cache_dir=project_root / "visualization_cache",
		data_dir=settings.__dict__["data-path"],
		dataset=dataset_name,
		types=("frozen-backbone",),
	)
	if written:
		print(f"External visualization cache updated for frozen {dataset_name}: {len(written)} file(s)", flush=True)


def execute_settings(settings):
	current_dataset = None
	for exp_id, config in enumerate(iter_configs(settings), start=1):
		if current_dataset is None:
			current_dataset = config["dataset"]
		elif config["dataset"] != current_dataset:
			maybe_build_dataset_visualization_cache(current_dataset, settings)
			current_dataset = config["dataset"]

		print_config(exp_id, config)
		if settings.__dict__["dry-run"]:
			print_cycle_plan(config)
			continue
		if settings.__dict__["parallel-cycles"]:
			run_config_cycles(config, max_workers=settings.__dict__["cycle-workers"])
		else:
			cycles = pending_cycles(config)
			if not cycles:
				print("All cycles already complete, skipping.", flush=True)
				continue
			for cycle_index, reason in cycles:
				print(f"Cycle {cycle_index + 1:02d} scheduled | seed={cycle_seed(config, cycle_index)} | {reason}", flush=True)
				run_cycle(config, cycle_index)
		print("\nDONE\n", flush=True)
	if not settings.__dict__["dry-run"] and current_dataset is not None:
		maybe_build_dataset_visualization_cache(current_dataset, settings)
