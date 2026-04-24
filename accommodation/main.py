import argparse
import torch

from accommodation.experiments.experiments.accommodation.agnews import run_accommodation_agnews_experiment
from accommodation.experiments.experiments.accommodation.cifar import run_accommodation_cifar_experiment
from accommodation.experiments.experiments.accommodation.fmnist import run_accommodation_fashion_mnist_experiment
from accommodation.experiments.experiments.accommodation.imdb import run_accommodation_imdb_experiment
from accommodation.experiments.experiments.accommodation.mnist import run_accommodation_mnist_experiment
from accommodation.experiments.experiments.accommodation.sctp import run_accommodation_sctp_experiment
from accommodation.experiments.experiments.accommodation.yelp import run_accommodation_yelp_experiment
from accommodation.experiments.experiments.linear.agnews import run_linear_agnews_experiment
from accommodation.experiments.experiments.linear.cifar import run_linear_cifar_experiment
from accommodation.experiments.experiments.linear.fmnist import run_linear_fashion_mnist_experiment
from accommodation.experiments.experiments.linear.imdb import run_linear_imdb_experiment
from accommodation.experiments.experiments.linear.mnist import run_linear_mnist_experiment
from accommodation.experiments.experiments.linear.sctp import run_linear_sctp_experiment
from accommodation.experiments.experiments.linear.yelp import run_linear_yelp_experiment

def parse_args():
    parser = argparse.ArgumentParser(description="MNIST Linear Classifier Experiment")
    parser.add_argument("--num-cycles", dest="num-cycles", type=int, default=2, help="Number of training cycles (default: 2)")
    parser.add_argument("--type", dest="type", type=str, choices=["linear", "accommodation"], default="accommodation", help="Type of experiment: 'linear' or 'accommodation' (default: accommodation)")
    parser.add_argument("--device", dest="device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use: 'cuda' or 'cpu' (default: auto)")
    parser.add_argument("--base-seed", dest="base-seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--dataset", dest="dataset", type=str, choices=["MNIST", "FMNIST", "CIFAR10", "IMDB", "SCTP", "AGNEWS", "YELP"], required=True, help="Dataset to use (required)")
    parser.add_argument("--data-path", dest="data-path", type=str, default="./datasets", help="Path to datasets (default: ./datasets)")
    parser.add_argument("--epochs", type=int, default=10, required=True, help="Number of training epochs (required)")
    parser.add_argument("--input-dim", dest="input-dim", type=int, default=32, help="Input dimension size (default: 32)")
    parser.add_argument("--embedding-dim", dest="embedding-dim", type=int, default=32, help="Embedding dimension size (default: 32)")
    parser.add_argument("--hidden-dim", dest="hidden-dim", type=int, default=512, help="Hidden layer dimension (default: 512)")
    parser.add_argument("--num-classes", dest="num-classes", type=int, required=True, help="Number of classes (required)")
    parser.add_argument("--num-potents-per-class", dest="num-potents-per-class", type=int, default=5, help="Potent units per class (default: 5)")
    parser.add_argument("--neutral-potents", dest="neutral-potents", type=int, default=10, help="Neutral potent units (default: 10)")
    parser.add_argument("--negative-potents", dest="negative-potents", type=bool, default=True, help="Use negative potents (default: True)")
    parser.add_argument("--latent-dim", dest="latent-dim", type=int, default=30, help="Latent dimension size (default: 30)")
    parser.add_argument("--plasticity", dest="plasticity", type=bool, default=True, help="Enable plasticity (default: True)")
    parser.add_argument("--plasticity-gamma", dest="plasticity-gamma", type=float, default=5, help="Plasticity gamma parameter (default: 5)")
    parser.add_argument("--differentiation-lambda", dest="differentiation-lambda", type=float, default=0.5, help="Differentiation lambda parameter (default: 0.5)")
    parser.add_argument("--results-dir", dest="results-dir", type=str, default="./results/", help="Directory to save results (default: ./results/)")
    return vars(parser.parse_args())


def run_linear_experiment(args):
    match args['dataset']:
        case "MNIST": run_linear_mnist_experiment(args)
        case "FMNIST": run_linear_fashion_mnist_experiment(args)
        case "CIFAR10": run_linear_cifar_experiment(args)
        case "IMDB": run_linear_imdb_experiment(args)
        case "SCTP": run_linear_sctp_experiment(args)
        case "AGNEWS": run_linear_agnews_experiment(args)
        case "YELP": run_linear_yelp_experiment(args)
        case _: print("Unknown dataset")


def run_accommodation_experiment(args):
    match args['dataset']:
        case "MNIST": run_accommodation_mnist_experiment(args)
        case "FMNIST": run_accommodation_fashion_mnist_experiment(args)
        case "CIFAR10": run_accommodation_cifar_experiment(args)
        case "IMDB": run_accommodation_imdb_experiment(args)
        case "SCTP": run_accommodation_sctp_experiment(args)
        case "AGNEWS": run_accommodation_agnews_experiment(args)
        case "YELP": run_accommodation_yelp_experiment(args)
        case _: print("Unknown dataset")


def main():
    args = parse_args()
    if args['type'] == 'linear': run_linear_experiment(args)
    else: run_accommodation_experiment(args)
