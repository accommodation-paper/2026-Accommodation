import torch
from torch.utils.data import DataLoader, random_split

from accommodation.experiments.classifiers.gru_linear_classifier import instantiate_gru_linear_classifier
from accommodation.experiments.runners.linear.text_model_runner import run_experiment
from accommodation.datasets.text.agnews import AGNewsDataset
from accommodation.model.set_seed import set_seed

def run_linear_agnews_experiment(args):
    set_seed(42)
    dataset = AGNewsDataset(args['data-path'])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    run_experiment("AGNews", args, dataset.vocab_size, train_loader, val_loader, instantiate_gru_linear_classifier)