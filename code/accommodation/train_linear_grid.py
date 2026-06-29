from accommodation.experiments.experiments.linear.cifar import run_linear_cifar_experiment
from accommodation.experiments.experiments.linear.fmnist import run_linear_fashion_mnist_experiment
from accommodation.experiments.experiments.linear.agnews import run_linear_agnews_experiment
from accommodation.experiments.experiments.linear.imdb import run_linear_imdb_experiment
from accommodation.experiments.experiments.linear.mnist import run_linear_mnist_experiment
from accommodation.experiments.experiments.linear.yelp import run_linear_yelp_experiment

from accommodation import train_accommodation_grid as grid_runner


DATASET_SPECS = {
	"MNIST": {"num-classes": 10},
	"FMNIST": {"num-classes": 10},
	"CIFAR10": {"num-classes": 10, "embedding-dim": 256},
	"AGNEWS": {"num-classes": 4},
	"IMDB": {"num-classes": 2},
	"YELP": {"num-classes": 2},
}


def run_linear_experiment(args):
	match args["dataset"]:
		case "MNIST":
			run_linear_mnist_experiment(args)
		case "FMNIST":
			run_linear_fashion_mnist_experiment(args)
		case "CIFAR10":
			run_linear_cifar_experiment(args)
		case "AGNEWS":
			run_linear_agnews_experiment(args)
		case "IMDB":
			run_linear_imdb_experiment(args)
		case "YELP":
			run_linear_yelp_experiment(args)
		case _:
			raise KeyError(f"Unsupported linear dataset: {args['dataset']}")


def create_settings(**overrides):
	defaults = {
		**grid_runner.DEFAULTS,
		"type": "linear",
		"datasets": list(DATASET_SPECS.keys()),
		"rules": grid_runner.LINEAR_RULES,
	}
	return grid_runner.create_settings(**{**defaults, **overrides})


def build_base_config(settings, dataset_name):
	spec = DATASET_SPECS[dataset_name]
	return {
		**grid_runner.DEFAULTS,
		"dataset": dataset_name,
		"type": "linear",
		"device": grid_runner.default_device() if getattr(settings, "auto-device", False) else settings.device,
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
	for dataset_name in settings.datasets:
		base_config = build_base_config(settings, dataset_name)
		config = dict(base_config)
		config["results-dir"] = f"{settings.__dict__['results-dir']}/linear"
		yield config


def execute_settings(settings):
	original_specs = grid_runner.DATASET_SPECS
	original_run_one = grid_runner.run_one
	try:
		grid_runner.DATASET_SPECS = DATASET_SPECS
		grid_runner.run_one = run_linear_experiment
		for exp_id, config in enumerate(iter_configs(settings), start=1):
			grid_runner.print_config(exp_id, config)
			if settings.__dict__["dry-run"]:
				grid_runner.print_cycle_plan(config)
				continue
			if settings.__dict__["parallel-cycles"]:
				grid_runner.run_config_cycles(config, max_workers=settings.__dict__["cycle-workers"])
			else:
				cycles = grid_runner.pending_cycles(config)
				if not cycles:
					print("All cycles already complete, skipping.", flush=True)
					continue
				for cycle_index, reason in cycles:
					print(
						f"Cycle {cycle_index + 1:02d} scheduled | seed={grid_runner.cycle_seed(config, cycle_index)} | {reason}",
						flush=True,
					)
					grid_runner.run_cycle(config, cycle_index)
			print("\nDONE\n", flush=True)
	finally:
		grid_runner.DATASET_SPECS = original_specs
		grid_runner.run_one = original_run_one
