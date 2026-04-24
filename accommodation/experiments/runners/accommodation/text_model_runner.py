import os
from typing import Callable, Any

import torch
from torch.utils.data import DataLoader

import torch.nn.functional as F

from accommodation.experiments.helpers.result_serializer import ResultSerializer
from accommodation.experiments.helpers.results_accessor import result_path, save
from accommodation.experiments.helpers._utils import create_configuration
from accommodation.model.accommodation_layer import Policy
from accommodation.model.accommodation_loss import AccommodationLoss
from accommodation.model.field import Field

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from accommodation.model.set_seed import set_seed


def run_experiment(dataset: str, args: dict, vocab_size: int, train_loader: DataLoader, val_loader: DataLoader, instantiate_model: Callable):
    for cycle in range(args["num-cycles"]):
        seed = args['base-seed'] + cycle
        set_seed(seed)
        serializer = ResultSerializer()
        model = instantiate_model(args, vocab_size).to(args['device'])
        serializable_configuration = config(args)

        criterion = AccommodationLoss(differentiation_lambda=args['differentiation-lambda'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        metrics = []
        fields = [Field(model.accommodation_layer)]
        model.accommodation_layer.reset_plasticity_stats()
        for epoch in range(1, args['epochs'] + 1):
            model.train()
            train_loss = 0.0

            model.accommodation_layer.reset_plasticity_stats()
            for xb, yb in train_loader:
                xb, yb = xb.to(args['device']), yb.to(args['device'])
                attention_mask = (xb != 0).long()
                optimizer.zero_grad()
                h = model.encode(xb, attention_mask)
                logits, differentiation_tensor = model.accommodation_layer(h)
                loss = criterion(differentiation_tensor, logits, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                model.accommodation_layer.add_info(h.detach(), yb)
            model.accommodation_layer.update_plasticity(gamma=args['plasticity-gamma'], p_min=0, p_max=1)
            train_loss /= max(1, len(train_loader))

            model.eval()
            val_loss = 0.0

            all_probs, all_preds, all_labels = [], [], []
            nll_sum = 0.0
            n_total = 0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(args['device']), yb.to(args['device'])

                    attention_mask = (xb != 0).long()
                    logits, differentiation_tensor = model(xb, attention_mask)
                    loss = criterion(differentiation_tensor, logits, yb)
                    val_loss += loss.item()

                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(logits, dim=1)

                    all_probs.append(probs.cpu())
                    all_preds.append(preds.cpu())
                    all_labels.append(yb.cpu())

                    nll_sum += F.cross_entropy(logits, yb, reduction="sum").item()
                    n_total += yb.size(0)

            policy = serializable_configuration["computation"]["policy"]
            neutral = serializable_configuration["field"]["neutral_potents"]
            negpot = serializable_configuration["field"]["negative_potents"]
            pospot = serializable_configuration["field"]["positive_potents"]

            out_path = os.path.join(
                args['results-dir'],
                args['type'],
                args['dataset'],
                str(policy),
                f"POS{pospot}-NEU{neutral}-NEG{negpot}",
                str(seed),
            )
            os.makedirs(out_path, exist_ok=True)

            model_path = os.path.join(out_path, f"snapshot_{epoch:02d}.pt")
            torch.save(model.state_dict(), model_path)

            val_loss /= max(1, len(val_loader))

            all_probs = torch.cat(all_probs).numpy()
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            val_acc = accuracy_score(all_labels, all_preds)

            val_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
            val_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
            val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            val_nll = nll_sum / max(1, n_total)

            metrics.append({
                "accuracy": float(val_acc),
                "precision_macro": float(val_prec),
                "recall_macro": float(val_rec),
                "f1_macro": float(val_f1),
                "val_nll": float(val_nll)
            })
            fields.append(Field(model.accommodation_layer))

        record = serializer.serialize(
            seed=seed,
            dataset=dataset,
            fields=fields,
            num_classes=args['num-classes'],
            epochs=args['epochs'],
            metrics=metrics,
            configuration=serializable_configuration,
            cycle=cycle + 1
        )

        out_path = result_path("accommodation", args, seed, cycle)
        save(out_path, record)

        print(f"Cycle {cycle + 1:02d} Done | Results saved on: {out_path}")


def config(args: dict) -> dict[str, dict[str, int | Any] | dict[str, str]]:
    return create_configuration(
        args['embedding-dim'],
        Policy.Likelihood,
        args["neutral-potents"],
        args["num-potents-per-class"],
        args['latent-dim'],
        args["negative-potents"],
        args['num-classes'],)