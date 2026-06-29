from torch.utils.data import DataLoader
from torchvision import transforms

from accommodation.datasets.vision.cifar10 import Cifar10Dataset
from accommodation.experiments.classifiers.cifar_accommodation_classifier import instantiate_cifar_accommodation_classifier
from accommodation.experiments.runners.accommodation.image_model_runner import run_experiment
from accommodation.model.set_seed import set_seed


def run_accommodation_cifar_experiment(args):
    set_seed(42)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    train_dataset = Cifar10Dataset(args['data-path'], train=True, transform=train_transform)
    val_dataset = Cifar10Dataset(args['data-path'], train=False, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    run_experiment(
        dataset="CIFAR10",
        args=args,
        train_loader=train_loader,
        val_loader=val_loader,
        instantiate_model=instantiate_cifar_accommodation_classifier)