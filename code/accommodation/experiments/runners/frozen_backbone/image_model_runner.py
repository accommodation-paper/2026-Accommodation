import json
import os
from pathlib import Path
from typing import Callable

import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from accommodation.experiments.helpers._utils import create_configuration
from accommodation.experiments.helpers.result_serializer import ResultSerializer
from accommodation.experiments.helpers.results_accessor import result_path, save
from accommodation.model.accommodation_layer import Policy
from accommodation.model.accommodation_loss import AccommodationLoss
from accommodation.model.field import Field
from accommodation.model.set_seed import set_seed


def run_experiment(
	dataset: str,
	args: dict,
	train_loader: DataLoader,
	val_loader: DataLoader,
	instantiate_backbone_model: Callable,
	instantiate_head_model: Callable,
):
	for cycle in range(args["num-cycles"]):
		seed = args["base-seed"] + cycle
		set_seed(seed)
		serializer = ResultSerializer()

		backbone_model, backbone_metrics = _train_backbone_stage(
			args=args,
			train_loader=train_loader,
			val_loader=val_loader,
			instantiate_backbone_model=instantiate_backbone_model,
			seed=seed,
			cycle=cycle,
		)

		model = instantiate_head_model(args).to(args["device"])
		model.backbone.load_state_dict(backbone_model.backbone.state_dict())
		for parameter in model.backbone.parameters():
			parameter.requires_grad = False
		model.backbone.eval()

		serializable_configuration = _config(args)
		serializable_configuration["training"] = {
			"mode": "frozen-backbone",
			"backbone_epochs": args["backbone-epochs"],
			"head_epochs": args["epochs"],
			"frozen_backbone": True,
		}
		serializable_configuration["backbone_pretraining"] = backbone_metrics

		criterion = AccommodationLoss(differentiation_lambda=args["differentiation-lambda"])
		optimizer = torch.optim.Adam(
			(parameter for parameter in model.parameters() if parameter.requires_grad),
			lr=1e-3,
		)

		metrics = []
		fields = [Field(model.accommodation_layer)]
		model.accommodation_layer.reset_plasticity_stats()
		for epoch in range(1, args["epochs"] + 1):
			model.train()
			model.backbone.eval()
			for parameter in model.backbone.parameters():
				parameter.requires_grad = False
			train_loss = 0.0
			for xb, yb in train_loader:
				xb, yb = xb.to(args["device"]), yb.to(args["device"])
				optimizer.zero_grad()
				with torch.no_grad():
					h = model.backbone(xb)
				logits, differentiation_tensor = model.accommodation_layer(h)
				loss = criterion(differentiation_tensor, logits, yb)
				loss.backward()
				optimizer.step()
				train_loss += loss.item()
				model.accommodation_layer.add_info(h.detach(), yb)
			model.accommodation_layer.update_plasticity(gamma=args["plasticity-gamma"], p_min=0, p_max=1)
			train_loss /= len(train_loader)

			model.eval()
			val_loss = 0.0
			all_probs, all_preds, all_labels = [], [], []
			with torch.no_grad():
				for xb, yb in val_loader:
					xb, yb = xb.to(args["device"]), yb.to(args["device"])
					logits, differentiation_tensor = model(xb)
					loss = criterion(differentiation_tensor, logits, yb)
					val_loss += loss.item()

					probs = torch.softmax(logits, dim=1)
					preds = torch.argmax(logits, dim=1)
					all_probs.append(probs.cpu())
					all_preds.append(preds.cpu())
					all_labels.append(yb.cpu())

			policy = serializable_configuration["computation"]["policy"]
			neutral = serializable_configuration["field"]["neutral_potents"]
			negpot = serializable_configuration["field"]["negative_potents"]
			pospot = serializable_configuration["field"]["positive_potents"]
			snapshot_dir = os.path.join(
				args["results-dir"],
				args["type"],
				args["dataset"],
				str(policy),
				f"POS{pospot}-NEU{neutral}-NEG{negpot}",
				str(seed),
			)
			os.makedirs(snapshot_dir, exist_ok=True)
			torch.save(model.state_dict(), os.path.join(snapshot_dir, f"snapshot_{epoch:02d}.pt"))

			val_loss /= len(val_loader)
			all_probs = torch.cat(all_probs, dim=0).numpy()
			all_preds = torch.cat(all_preds, dim=0).numpy()
			all_labels = torch.cat(all_labels, dim=0).numpy()

			fields.append(Field(model.accommodation_layer))
			metrics.append({
				"accuracy": float(accuracy_score(all_labels, all_preds)),
				"precision_macro": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
				"recall_macro": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
				"f1_macro": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
				"precision_weighted": float(precision_score(all_labels, all_preds, average="weighted", zero_division=0)),
				"recall_weighted": float(recall_score(all_labels, all_preds, average="weighted", zero_division=0)),
				"f1_weighted": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
				"log_loss": float(log_loss(all_labels, all_probs, labels=list(range(args["num-classes"])))),
				"train_loss": float(train_loss),
				"val_loss": float(val_loss),
			})

		record = serializer.serialize(
			seed=seed,
			dataset=dataset,
			fields=fields,
			num_classes=args["num-classes"],
			epochs=args["epochs"],
			metrics=metrics,
			configuration=serializable_configuration,
			cycle=cycle + 1,
		)
		record["visualization"] = None
		out_path = result_path(args["type"], args, seed, cycle)
		save(out_path, record)
		print(f"Cycle {cycle + 1:02d} Done | Frozen-backbone results saved on: {out_path}")


def _train_backbone_stage(args, train_loader, val_loader, instantiate_backbone_model, seed, cycle):
	model = instantiate_backbone_model(args).to(args["device"])
	criterion = CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	best_metrics = {
		"epoch": None,
		"accuracy": -1.0,
		"precision_macro": -1.0,
		"recall_macro": -1.0,
		"f1_macro": -1.0,
	}
	for epoch in range(1, args["backbone-epochs"] + 1):
		model.train()
		for xb, yb in train_loader:
			xb, yb = xb.to(args["device"]), yb.to(args["device"])
			optimizer.zero_grad()
			logits = model(xb)
			loss = criterion(logits, yb)
			loss.backward()
			optimizer.step()

		model.eval()
		all_preds, all_labels = [], []
		with torch.no_grad():
			for xb, yb in val_loader:
				xb, yb = xb.to(args["device"]), yb.to(args["device"])
				logits = model(xb)
				preds = torch.argmax(logits, dim=1)
				all_preds.append(preds.cpu())
				all_labels.append(yb.cpu())

		all_preds = torch.cat(all_preds).numpy()
		all_labels = torch.cat(all_labels).numpy()
		val_acc = accuracy_score(all_labels, all_preds)
		if float(val_acc) > best_metrics["accuracy"]:
			best_metrics = {
				"epoch": epoch,
				"accuracy": float(val_acc),
				"precision_macro": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
				"recall_macro": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
				"f1_macro": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
			}

	backbone_dir = Path(args["results-dir"]) / args["type"] / args["dataset"] / "pretrained_backbones" / str(seed)
	backbone_dir.mkdir(parents=True, exist_ok=True)
	backbone_state_path = backbone_dir / f"cycle_{cycle:02d}.pt"
	torch.save(model.state_dict(), backbone_state_path)
	metadata = {
		"seed": seed,
		"cycle": cycle + 1,
		"epochs": args["backbone-epochs"],
		"best_metrics": best_metrics,
		"state_dict_path": str(backbone_state_path),
	}
	with open(backbone_dir / f"cycle_{cycle:02d}.json", "w", encoding="utf-8") as file:
		json.dump(metadata, file, indent=2)
	print(
		f"Cycle {cycle + 1:02d} Backbone pretraining | "
		f"Best Epoch {best_metrics['epoch']:02d} | "
		f"Acc {best_metrics['accuracy']:.4f}"
	)
	return model, metadata


def _config(args: dict) -> dict:
	return create_configuration(
		args["embedding-dim"],
		Policy.Likelihood,
		args["neutral-potents"],
		args["num-potents-per-class"],
		args["latent-dim"],
		args["negative-potents"],
		args["num-classes"],
	)
