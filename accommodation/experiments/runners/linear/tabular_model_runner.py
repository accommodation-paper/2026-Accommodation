import json
import os
from typing import Callable

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_score,
    recall_score,
    f1_score, average_precision_score, balanced_accuracy_score, roc_auc_score,
)

from accommodation.model.set_seed import set_seed


def run_experiment(dataset: str, args: dict, train_loader: DataLoader, val_loader: DataLoader, instantiate_model: Callable):
    for cycle in range(args["num-cycles"]):
        seed = args['base-seed'] + cycle
        set_seed(seed)

        model = instantiate_model(args).to(args['device'])

        criterion = CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_metrics = {
            "epoch": None,
            "balanced_accuracy": -1
        }

        for epoch in range(1, 6):
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(args['device']), yb.to(args['device'])
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            all_probs, all_preds, all_labels = [], [], []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(args['device']), yb.to(args['device'])
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item()

                    probs = torch.softmax(logits, dim=1)[:, 1]
                    preds = torch.argmax(logits, dim=1)

                    all_probs.append(probs.cpu())
                    all_preds.append(preds.cpu())
                    all_labels.append(yb.cpu())

            val_loss /= len(val_loader)

            all_probs = torch.cat(all_probs).numpy()
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            val_auc = roc_auc_score(all_labels, all_probs)

            if float(balanced_accuracy_score(all_labels, all_preds)) > best_metrics["balanced_accuracy"]:
                best_metrics = {
                    "epoch": epoch,
                    "auc": float(val_auc),
                    "pr_auc": float(average_precision_score(all_labels, all_probs)),
                    "accuracy": float(accuracy_score(all_labels, all_preds)),
                    "balanced_accuracy": float(balanced_accuracy_score(all_labels, all_preds)),
                    "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
                    "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
                    "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
                    "log_loss": float(log_loss(all_labels, all_probs)),
                }

        record = {
            "dataset": dataset,
            "cycle": cycle + 1,
            "seed": seed,
            "epochs": 20,
            "best_epoch": best_metrics["epoch"],
            "metrics": best_metrics,
        }

        out_path = os.path.join(args['results-dir'], "linear", args['dataset'], str(seed))
        os.makedirs(out_path, exist_ok=True)

        with open(f"{out_path}/cycle_{cycle}.json", "w") as f:
            json.dump(record, f, indent=2)

        print(
            f"Cycle {cycle + 1:02d} | "
            f"Best Epoch {best_metrics['epoch']:02d} | "
            f"Acc {best_metrics['accuracy']:.4f} | "
        )

        model_path = out_path + f"/cycle_{cycle}.pt"
        torch.save(model.state_dict(), model_path)