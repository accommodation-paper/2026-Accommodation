import json
import os
from typing import Callable

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score, log_loss,
)

from accommodation.experiments.helpers.results_accessor import result_path
from accommodation.model.set_seed import set_seed


def run_experiment(dataset: str, args: dict, vocab_size: int, train_loader: DataLoader, val_loader: DataLoader, instantiate_model: Callable):
    for cycle in range(args["num-cycles"]):
        seed = args['base-seed'] + cycle
        set_seed(seed)
        model = instantiate_model(args, vocab_size).to(args['device'])

        criterion = CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_metrics = {
            "epoch": None,
            "auc_macro_ovr": -1.0,
            "accuracy": -1.0,
            "precision_macro": -1.0,
            "recall_macro": -1.0,
            "f1_macro": -1.0,
            "log_loss": 1e9,
        }

        for epoch in range(1, args['epochs'] + 1):
            model.train()
            train_loss = 0.0

            for xb, yb in train_loader:
                xb, yb = xb.to(args['device']), yb.to(args['device'])

                attention_mask = (xb != 0).long()

                optimizer.zero_grad()
                logits = model(xb, attention_mask)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= max(1, len(train_loader))

            model.eval()
            val_loss = 0.0

            all_probs, all_preds, all_labels = [], [], []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(args['device']), yb.to(args['device'])

                    attention_mask = (xb != 0).long()

                    logits = model(xb, attention_mask)
                    loss = criterion(logits, yb)
                    val_loss += loss.item()

                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(logits, dim=1)

                    all_probs.append(probs.cpu())
                    all_preds.append(preds.cpu())
                    all_labels.append(yb.cpu())

            val_loss /= max(1, len(val_loader))

            all_probs = torch.cat(all_probs).numpy()
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            val_acc = accuracy_score(all_labels, all_preds)
            val_ll = log_loss(all_labels, all_probs, labels=list(range(args['num-classes'])))

            val_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
            val_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
            val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

            if float(val_acc) > best_metrics["accuracy"]:
                best_metrics = {
                    "epoch": epoch,
                    "accuracy": float(val_acc),
                    "precision_macro": float(val_prec),
                    "recall_macro": float(val_rec),
                    "f1_macro": float(val_f1),
                    "log_loss": float(val_ll),
                }

        record = {
            "dataset": dataset,
            "cycle": cycle + 1,
            "seed": seed,
            "epochs": args['epochs'],
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
            f"F1(macro) {best_metrics['f1_macro']:.4f} | "
        )

        model_path = out_path + f"/cycle_{cycle}.pt"
        torch.save(model.state_dict(), model_path)