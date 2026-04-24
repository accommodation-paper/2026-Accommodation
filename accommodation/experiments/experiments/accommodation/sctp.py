import torch
from torch.utils.data import DataLoader, random_split

from accommodation.experiments.classifiers.mlp_accommodation_classifier import instantiate_mlp_accommodation_classifier
from accommodation.experiments.runners.accommodation.tabular_model_runner import run_experiment
from accommodation.datasets.tabular.sctp import SantanderCustomerTransactionDataset
from accommodation.model.set_seed import set_seed


def run_accommodation_sctp_experiment(args):
    set_seed(42)
    dataset = SantanderCustomerTransactionDataset(csv_path=args['data-path'])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    run_experiment(
        dataset="SCTP",
        args=args,
        train_loader=train_loader,
        val_loader=val_loader,
        instantiate_model=instantiate_mlp_accommodation_classifier)