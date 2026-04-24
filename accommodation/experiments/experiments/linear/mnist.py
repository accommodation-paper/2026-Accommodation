from torch.utils.data import DataLoader
from torchvision import transforms

from accommodation.experiments.classifiers.mnist_linear_classifier import instantiate_mnist_linear_classifier
from accommodation.experiments.runners.linear.image_model_runner import run_experiment
from accommodation.datasets.vision.mnist import MNISTDataset
from accommodation.model.set_seed import set_seed


def run_linear_mnist_experiment(args):
    set_seed(42)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNISTDataset(root=args['data-path'], train=True, transform=train_transform)
    val_dataset = MNISTDataset(root=args['data-path'], train=False, transform=test_transform)

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

    run_experiment("MNIST", args, train_loader, val_loader, instantiate_mnist_linear_classifier)