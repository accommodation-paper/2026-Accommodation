import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import torch

from accommodation.dataset_visualization_builder import build_visualization_cache
from accommodation.experiments.helpers.results_accessor import result_path
from accommodation.main import run_accommodation_experiment, run_linear_experiment


DEFAULTS = {
	"num-cycles": 10,
	"type": "accommodation",
	"device": "cuda",
	"auto-device": False,
	"base-seed": 42,
	"data-path": "data",
	"epochs": 80,
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


ACCOMMODATION_RULES = [
	{"plasticity": [False], "differentiation-lambda": [0], "num-potents-per-class": [5]},
	{"plasticity": [False], "differentiation-lambda": [0.5, 0.8, 0.9, 0.95], "num-potents-per-class": [5]},
	{"plasticity": [True], "plasticity-gamma": [1, 5, 10, 20], "differentiation-lambda": [0], "num-potents-per-class": [5]},
	{"plasticity": [True], "plasticity-gamma": [5], "differentiation-lambda": [0.5, 0.8, 0.9, 0.95], "num-potents-per-class": [5]},
	{"plasticity": [False], "differentiation-lambda": [0], "num-potents-per-class": [1, 5, 10, 20]},
	{"plasticity": [True], "plasticity-gamma": [5], "differentiation-lambda": [0.8], "num-potents-per-class": [1, 10, 20]},
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
	settings.setdefault("rules", ACCOMMODATION_RULES)
	return SimpleNamespace(**settings)


def expand(rule):
	if not rule:
		yield {}
		return
	keys = list(rule.keys())
	values = list(rule.values())
	for combo in product(*values):
		yield dict(zip(keys, combo))


def run_one(config):
	if config["type"] == "linear":
		run_linear_experiment(config)
	else:
		run_accommodation_experiment(config)


def cycle_seed(config, cycle_index):
	return config["base-seed"] + cycle_index


def cycle_candidates(cycle_index):
	if cycle_index == 0:
		return [0]
	return [0, cycle_index]


def linear_paths(config, cycle_index, seed):
	base = Path(config["results-dir"]) / "linear" / config["dataset"] / str(seed)
	return [
		(base / f"cycle_{candidate}.json", base / f"cycle_{candidate}.pt", base)
		for candidate in cycle_candidates(cycle_index)
	]


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


def valid_linear_record(json_path):
	try:
		with open(json_path, "r", encoding="utf-8") as file:
			record = json.load(file)
	except (OSError, json.JSONDecodeError):
		return False
	return bool(record.get("metrics")) and ("best_epoch" in record or "best-epoch" in record)


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

	if config["type"] == "linear":
		for json_path, model_path, _ in linear_paths(config, cycle_index, seed):
			if json_path.exists() and model_path.exists() and valid_linear_record(json_path):
				return True, None
		return False, "missing linear json/model"

	for path in summary_paths(config, cycle_index, seed):
		if not path.exists():
			continue
		trace = valid_serialized_trace(path, config["epochs"])
		if trace is not None:
			return True, None

	return False, "missing or invalid serialized trace"


def cleanup_cycle(config, cycle_index):
	seed = cycle_seed(config, cycle_index)

	if config["type"] == "linear":
		for json_path, model_path, base_dir in linear_paths(config, cycle_index, seed):
			if json_path.exists():
				json_path.unlink()
			if model_path.exists():
				model_path.unlink()
			if base_dir.exists():
				shutil.rmtree(base_dir, ignore_errors=True)
		return

	for path in summary_paths(config, cycle_index, seed):
		if path.exists():
			path.unlink()

	shutil.rmtree(snapshot_dir(config, seed), ignore_errors=True)


def run_cycle(config, cycle_index):
	cleanup_cycle(config, cycle_index)
	cycle_config = {
		**config,
		"num-cycles": 1,
		"base-seed": cycle_seed(config, cycle_index),
	}
	run_one(cycle_config)
	return cycle_index, cycle_config["base-seed"]


def config_name(config):
	if config["type"] == "linear":
		return "linear"
	return (
		f"PL{config.get('plasticity')}"
		f"_GM{config.get('plasticity-gamma')}"
		f"_DF{config.get('differentiation-lambda')}"
		f"_PPC{config.get('num-potents-per-class')}"
	)


def build_base_config(settings, dataset_name):
	spec = DATASET_SPECS[dataset_name]
	return {
		**DEFAULTS,
		"dataset": dataset_name,
		"type": settings.type,
		"device": default_device() if getattr(settings, "auto-device", False) else settings.device,
		"data-path": settings.__dict__["data-path"],
		"results-dir": settings.__dict__["results-dir"],
		"epochs": settings.epochs,
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
	rules = settings.rules
	for dataset_name in settings.datasets:
		base_config = build_base_config(settings, dataset_name)
		for rule in rules:
			for combo in expand(rule):
				config = {**base_config, **combo}
				config["results-dir"] = os.path.join(settings.__dict__["results-dir"], config_name(config))
				yield config


def print_config(exp_id, config):
	print("\n" + "=" * 90, flush=True)
	print(f"EXPERIMENT {exp_id}", flush=True)
	print(config["dataset"], flush=True)
	if config["type"] == "linear":
		print(f"type=linear | cycles={config.get('num-cycles')}", flush=True)
	else:
		print(
			f"plasticity={config.get('plasticity')} | "
			f"gamma={config.get('plasticity-gamma') if config.get('plasticity') else 0} | "
			f"diff={config.get('differentiation-lambda')} | "
			f"potents={config.get('num-potents-per-class')} | "
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
		print(
			f"Cycle {cycle_index + 1:02d} scheduled | seed={cycle_seed(config, cycle_index)} | {reason}",
			flush=True,
		)

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
		types=("accommodation",),
	)
	if written:
		print(f"External visualization cache updated for accommodation {dataset_name}: {len(written)} file(s)", flush=True)


def execute_settings(settings):
	current_dataset = None
	for exp_id, config in enumerate(iter_configs(settings), start=1):
		if settings.type == "accommodation":
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
				print(
					f"Cycle {cycle_index + 1:02d} scheduled | seed={cycle_seed(config, cycle_index)} | {reason}",
					flush=True,
				)
				run_cycle(config, cycle_index)
		print("\nDONE\n", flush=True)
	if settings.type == "accommodation" and not settings.__dict__["dry-run"] and current_dataset is not None:
		maybe_build_dataset_visualization_cache(current_dataset, settings)
