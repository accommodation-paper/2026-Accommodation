from torch.utils.data import DataLoader
from torchvision import transforms

from accommodation.experiments.classifiers.mnist_accommodation_classifier import instantiate_mnist_accommodation_classifier
from accommodation.experiments.runners.accommodation.image_model_runner import run_experiment
from accommodation.datasets.vision.fashion_mnist import FashionMNISTDataset
from accommodation.model.set_seed import set_seed


def run_accommodation_fashion_mnist_experiment(args):
    set_seed(42)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = FashionMNISTDataset(args['data-path'], train=True, transform=train_transform)
    val_dataset = FashionMNISTDataset(args['data-path'], train=False, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    run_experiment(
        dataset="FashionMNIST",
        args=args,
        train_loader=train_loader,
        val_loader=val_loader,
            instantiate_model=instantiate_mnist_accommodation_classifier)