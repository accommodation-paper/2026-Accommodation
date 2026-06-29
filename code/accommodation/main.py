from accommodation.experiments.experiments.accommodation.agnews import run_accommodation_agnews_experiment
from accommodation.experiments.experiments.accommodation.cifar import run_accommodation_cifar_experiment
from accommodation.experiments.experiments.accommodation.fmnist import run_accommodation_fashion_mnist_experiment
from accommodation.experiments.experiments.accommodation.imdb import run_accommodation_imdb_experiment
from accommodation.experiments.experiments.accommodation.mnist import run_accommodation_mnist_experiment
from accommodation.experiments.experiments.accommodation.yelp import run_accommodation_yelp_experiment
from accommodation.experiments.experiments.frozen_backbone.agnews import run_frozen_backbone_agnews_experiment
from accommodation.experiments.experiments.frozen_backbone.cifar import run_frozen_backbone_cifar_experiment
from accommodation.experiments.experiments.frozen_backbone.fmnist import run_frozen_backbone_fashion_mnist_experiment
from accommodation.experiments.experiments.frozen_backbone.imdb import run_frozen_backbone_imdb_experiment
from accommodation.experiments.experiments.frozen_backbone.mnist import run_frozen_backbone_mnist_experiment
from accommodation.experiments.experiments.frozen_backbone.yelp import run_frozen_backbone_yelp_experiment
from accommodation.experiments.experiments.linear.agnews import run_linear_agnews_experiment
from accommodation.experiments.experiments.linear.cifar import run_linear_cifar_experiment
from accommodation.experiments.experiments.linear.fmnist import run_linear_fashion_mnist_experiment
from accommodation.experiments.experiments.linear.imdb import run_linear_imdb_experiment
from accommodation.experiments.experiments.linear.mnist import run_linear_mnist_experiment
from accommodation.experiments.experiments.linear.yelp import run_linear_yelp_experiment

def run_linear_experiment(args):
    match args['dataset']:
        case "MNIST": run_linear_mnist_experiment(args)
        case "FMNIST": run_linear_fashion_mnist_experiment(args)
        case "CIFAR10": run_linear_cifar_experiment(args)
        case "IMDB": run_linear_imdb_experiment(args)
        case "AGNEWS": run_linear_agnews_experiment(args)
        case "YELP": run_linear_yelp_experiment(args)
        case _: print("Unknown dataset")


def run_accommodation_experiment(args):
    match args['dataset']:
        case "MNIST": run_accommodation_mnist_experiment(args)
        case "FMNIST": run_accommodation_fashion_mnist_experiment(args)
        case "CIFAR10": run_accommodation_cifar_experiment(args)
        case "IMDB": run_accommodation_imdb_experiment(args)
        case "AGNEWS": run_accommodation_agnews_experiment(args)
        case "YELP": run_accommodation_yelp_experiment(args)
        case _: print("Unknown dataset")


def run_frozen_backbone_experiment(args):
    match args['dataset']:
        case "AGNEWS": run_frozen_backbone_agnews_experiment(args)
        case "CIFAR10": run_frozen_backbone_cifar_experiment(args)
        case "FMNIST": run_frozen_backbone_fashion_mnist_experiment(args)
        case "IMDB": run_frozen_backbone_imdb_experiment(args)
        case "MNIST": run_frozen_backbone_mnist_experiment(args)
        case "YELP": run_frozen_backbone_yelp_experiment(args)
        case _: raise NotImplementedError(f"Frozen-backbone experiments are not implemented for dataset {args['dataset']}")


from accommodation import train_accommodation_grid as accommodation_grid
from accommodation import train_frozen_backbone_grid as frozen_backbone_grid
from accommodation import train_linear_grid as linear_grid


RUN_ACCOMMODATION = True
RUN_FROZEN_BACKBONE = True
RUN_LINEAR = True


ACCOMMODATION_SETTINGS = {
    "datasets": ["MNIST", "FMNIST", "CIFAR10", "AGNEWS", "IMDB", "YELP"],
    "device": accommodation_grid.DEFAULTS["device"],
    "auto-device": False,
    "data-path": accommodation_grid.DEFAULTS["data-path"],
    "results-dir": accommodation_grid.DEFAULTS["results-dir"],
    "epochs": accommodation_grid.DEFAULTS["epochs"],
    "num-cycles": accommodation_grid.DEFAULTS["num-cycles"],
    "base-seed": accommodation_grid.DEFAULTS["base-seed"],
    "embedding-dim": accommodation_grid.DEFAULTS["embedding-dim"],
    "hidden-dim": accommodation_grid.DEFAULTS["hidden-dim"],
    "latent-dim": accommodation_grid.DEFAULTS["latent-dim"],
    "neutral-potents": accommodation_grid.DEFAULTS["neutral-potents"],
    "negative-potents": accommodation_grid.DEFAULTS["negative-potents"],
    "parallel-cycles": True,
    "cycle-workers": accommodation_grid.DEFAULTS["cycle-workers"],
    "dry-run": False,
    "rules": accommodation_grid.ACCOMMODATION_RULES,
}


FROZEN_BACKBONE_SETTINGS = {
    "datasets": ["MNIST", "FMNIST", "CIFAR10", "AGNEWS", "IMDB", "YELP"],
    "device": frozen_backbone_grid.DEFAULTS["device"],
    "auto-device": False,
    "data-path": frozen_backbone_grid.DEFAULTS["data-path"],
    "results-dir": frozen_backbone_grid.DEFAULTS["results-dir"],
    "epochs": frozen_backbone_grid.DEFAULTS["epochs"],
    "backbone-epochs": frozen_backbone_grid.DEFAULTS["backbone-epochs"],
    "num-cycles": frozen_backbone_grid.DEFAULTS["num-cycles"],
    "base-seed": frozen_backbone_grid.DEFAULTS["base-seed"],
    "embedding-dim": frozen_backbone_grid.DEFAULTS["embedding-dim"],
    "hidden-dim": frozen_backbone_grid.DEFAULTS["hidden-dim"],
    "latent-dim": frozen_backbone_grid.DEFAULTS["latent-dim"],
    "neutral-potents": frozen_backbone_grid.DEFAULTS["neutral-potents"],
    "negative-potents": frozen_backbone_grid.DEFAULTS["negative-potents"],
    "parallel-cycles": True,
    "cycle-workers": frozen_backbone_grid.DEFAULTS["cycle-workers"],
    "dry-run": False,
    "rules": frozen_backbone_grid.RULES,
}


LINEAR_SETTINGS = {
    "datasets": ["MNIST", "FMNIST", "CIFAR10", "AGNEWS", "IMDB", "YELP"],
    "device": linear_grid.grid_runner.DEFAULTS["device"],
    "auto-device": False,
    "data-path": linear_grid.grid_runner.DEFAULTS["data-path"],
    "results-dir": linear_grid.grid_runner.DEFAULTS["results-dir"],
    "epochs": linear_grid.grid_runner.DEFAULTS["epochs"],
    "num-cycles": linear_grid.grid_runner.DEFAULTS["num-cycles"],
    "base-seed": linear_grid.grid_runner.DEFAULTS["base-seed"],
    "embedding-dim": linear_grid.grid_runner.DEFAULTS["embedding-dim"],
    "hidden-dim": linear_grid.grid_runner.DEFAULTS["hidden-dim"],
    "latent-dim": linear_grid.grid_runner.DEFAULTS["latent-dim"],
    "neutral-potents": linear_grid.grid_runner.DEFAULTS["neutral-potents"],
    "negative-potents": linear_grid.grid_runner.DEFAULTS["negative-potents"],
    "parallel-cycles": True,
    "cycle-workers": linear_grid.grid_runner.DEFAULTS["cycle-workers"],
    "dry-run": False,
}


def main():
    if RUN_ACCOMMODATION:
        print("\n### ACCOMMODATION GRID ###\n", flush=True)
        accommodation_grid.execute_settings(accommodation_grid.create_settings(**ACCOMMODATION_SETTINGS))

    if RUN_FROZEN_BACKBONE:
        print("\n### FROZEN BACKBONE GRID ###\n", flush=True)
        frozen_backbone_grid.execute_settings(frozen_backbone_grid.create_settings(**FROZEN_BACKBONE_SETTINGS))

    if RUN_LINEAR:
        print("\n### LINEAR GRID ###\n", flush=True)
        linear_grid.execute_settings(linear_grid.create_settings(**LINEAR_SETTINGS))


if __name__ == "__main__":
    main()
