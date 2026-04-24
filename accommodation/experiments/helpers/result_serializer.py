import torch

from accommodation.model.field import Field


def flat_map(l):
	result = []
	for item in l:
		result.extend(item)
	return result

def potent_pearson_correlation(field: Field):
	mu_list = [field.positive_mu]
	sigma_list = [field.positive_sigma]

	if field.negative_mu is not None:
		mu_list.append(field.negative_mu)
		sigma_list.append(field.negative_sigma)

	if field.neutral_mu is not None:
		mu_list.append(field.neutral_mu)
		sigma_list.append(field.neutral_sigma)

	mu_all = torch.cat(mu_list, dim=0)
	sigma_all = torch.cat(sigma_list, dim=0)
	mu_flat = mu_all.reshape(-1, mu_all.size(-1))
	sigma_flat = sigma_all.reshape(-1, sigma_all.size(-1))
	potent_vec = torch.cat([mu_flat, sigma_flat], dim=1)
	corr = torch.corrcoef(potent_vec)
	corr = torch.nan_to_num(corr, nan=0.0)
	return corr


class ResultSerializer:
	def serialize(
			self,
			seed: int,
			dataset: str,
			fields: list[Field],
			num_classes: int,
			epochs: int,
			metrics: list[dict[str, float]],
			configuration: dict | None = None,
			cycle: int | None = None,
	) -> dict:
		if len(fields) != epochs + 1: raise RuntimeError(f"fields must have length epochs ({epochs}) + 1, got {len(fields)}")
		if len(metrics) != epochs: raise RuntimeError(f"metrics must have length epochs ({epochs}), got {len(metrics)}")
		result = {
			"dataset": dataset,
			"seed": seed,
			"cycle": cycle,
			"num_classes": num_classes,
			"configuration": configuration,
			"epochs": [{
				"epoch": 0,
				"metrics": None,
				"potents": [p.to_dict() for p in fields[0].potents()]
			}
			]
		}
		for epoch in range(1, epochs + 1):
			epoch_dict = {
				"epoch": epoch,
				"metrics": metrics[epoch - 1],
				"potents": [p.to_dict() for p in fields[epoch].potents()]
			}
			result["epochs"].append(epoch_dict)
		return result

	def potents(self, field: Field) -> list[dict[str, int]]:
		return [{"id": idx, "polarity": field.potents()[idx].type.name, "class": field.potents()[idx].clazz} for idx in range(len(field.potents()))]