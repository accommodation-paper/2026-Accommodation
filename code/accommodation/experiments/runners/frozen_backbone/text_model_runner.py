import json
import os
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
	vocab_size: int,
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
			vocab_size=vocab_size,
			train_loader=train_loader,
			val_loader=val_loader,
			instantiate_backbone_model=instantiate_backbone_model,
			seed=seed,
			cycle=cycle,
		)

		model = instantiate_head_model(args, vocab_size).to(args["device"])
		model.embedding.load_state_dict(backbone_model.embedding.state_dict())
		model.gru.load_state_dict(backbone_model.gru.state_dict())
		model.proj.load_state_dict(backbone_model.proj.state_dict())
		for module in (model.embedding, model.gru, model.proj):
			for parameter in module.parameters():
				parameter.requires_grad = False
		model.embedding.eval()
		model.gru.eval()
		model.proj.eval()

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
			model.embedding.eval()
			model.gru.eval()
			model.proj.eval()
			train_loss = 0.0
			model.accommodation_layer.reset_plasticity_stats()
			for xb, yb in train_loader:
				xb, yb = xb.to(args["device"]), yb.to(args["device"])
				attention_mask = (xb != 0).long()
				optimizer.zero_grad()
				with torch.no_grad():
					h = model.encode(xb, attention_mask)
				logits, differentiation_tensor = model.accommodation_layer(h)
				loss = criterion(differentiation_tensor, logits, yb)
				loss.backward()
				optimizer.step()
				train_loss += loss.item()
				model.accommodation_layer.add_info(h.detach(), yb)
			model.accommodation_layer.update_plasticity(gamma=args["plasticity-gamma"], p_min=0, p_max=1)
			train_loss /= max(1, len(train_loader))

			model.eval()
			val_loss = 0.0
			all_probs, all_preds, all_labels = [], [], []
			nll_sum = 0.0
			n_total = 0
			with torch.no_grad():
				for xb, yb in val_loader:
					xb, yb = xb.to(args["device"]), yb.to(args["device"])
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

			all_preds = torch.cat(all_preds).numpy()
			all_labels = torch.cat(all_labels).numpy()
			fields.append(Field(model.accommodation_layer))
			metrics.append({
				"accuracy": float(accuracy_score(all_labels, all_preds)),
				"precision_macro": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
				"recall_macro": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
				"f1_macro": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
				"val_nll": float(nll_sum / max(1, n_total)),
				"train_loss": float(train_loss),
				"val_loss": float(val_loss / max(1, len(val_loader))),
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


def _train_backbone_stage(args, vocab_size, train_loader, val_loader, instantiate_backbone_model, seed, cycle):
	model = instantiate_backbone_model(args, vocab_size).to(args["device"])
	criterion = CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	best_metrics = {"epoch": None, "accuracy": -1.0, "precision_macro": -1.0, "recall_macro": -1.0, "f1_macro": -1.0}
	for epoch in range(1, args["backbone-epochs"] + 1):
		model.train()
		for xb, yb in train_loader:
			xb, yb = xb.to(args["device"]), yb.to(args["device"])
			attention_mask = (xb != 0).long()
			optimizer.zero_grad()
			logits = model(xb, attention_mask)
			loss = criterion(logits, yb)
			loss.backward()
			optimizer.step()
		model.eval()
		all_preds, all_labels = [], []
		with torch.no_grad():
			for xb, yb in val_loader:
				xb, yb = xb.to(args["device"]), yb.to(args["device"])
				attention_mask = (xb != 0).long()
				preds = torch.argmax(model(xb, attention_mask), dim=1)
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
