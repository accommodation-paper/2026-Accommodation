from torch.utils.data import DataLoader
from torchvision import transforms

from accommodation.datasets.vision.fashion_mnist import FashionMNISTDataset
from accommodation.experiments.classifiers.mnist_accommodation_classifier import instantiate_mnist_accommodation_classifier
from accommodation.experiments.classifiers.mnist_linear_classifier import instantiate_mnist_linear_classifier
from accommodation.experiments.runners.frozen_backbone.image_model_runner import run_experiment
from accommodation.model.set_seed import set_seed


def run_frozen_backbone_fashion_mnist_experiment(args):
	set_seed(42)
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
	train_dataset = FashionMNISTDataset(args["data-path"], train=True, transform=transform)
	val_dataset = FashionMNISTDataset(args["data-path"], train=False, transform=transform)
	train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
	val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
	run_experiment("FashionMNIST", args, train_loader, val_loader, instantiate_mnist_linear_classifier, instantiate_mnist_accommodation_classifier)
