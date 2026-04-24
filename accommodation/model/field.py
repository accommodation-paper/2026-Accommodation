from enum import Enum

import torch
from accommodation.model.accommodation_layer import AccommodationLayer
import torch.nn.functional as F


class PotentType(Enum):
	Positive = 1
	Negative = 2
	Neutral = 3


class Potent:
	def __init__(self, id_: int, means: list[float], stds: list[float], plasticity: float|None, type: PotentType, clazz: int|None):
		self.id = id_
		self.means = means
		self.stds = stds
		self.plasticity = plasticity
		self.type = type
		self.clazz = clazz

	def to_dict(self):
		return {
			"id": self.id,
			"means": self.means,
			"stds": self.stds,
			"plasticity": self.plasticity,
			"type": self.type.name,
			"class": self.clazz
		}


def flat_map(l):
	result = []
	for item in l:
		result.extend(item)
	return result


class Field:
	def __init__(self, accommodation_layer: AccommodationLayer):
		self.positive_mu = torch.clone(accommodation_layer.positive_mu)
		self.positive_sigma = torch.clone(accommodation_layer.positive_sigma)
		self.positive_plasticity = torch.clone(accommodation_layer.positive_plasticity) if accommodation_layer.plasticity else None
		self.positive_specialization = accommodation_layer.positive_specialization if accommodation_layer.plasticity else None
		self.positive_usage = torch.clone(accommodation_layer.positive_usage) if accommodation_layer.plasticity else None
		self.negative_mu = torch.clone(accommodation_layer.negative_mu) if accommodation_layer.negative_potents else None
		self.negative_sigma = torch.clone(accommodation_layer.negative_sigma) if accommodation_layer.negative_potents else None
		self.negative_plasticity = torch.clone(accommodation_layer.negative_plasticity) if accommodation_layer.plasticity and accommodation_layer.negative_potents else None
		self.negative_specialization = accommodation_layer.negative_specialization if accommodation_layer.plasticity and accommodation_layer.negative_potents else None
		self.negative_usage = torch.clone(accommodation_layer.negative_usage) if accommodation_layer.plasticity and accommodation_layer.negative_potents else None
		self.neutral_mu = torch.clone(accommodation_layer.neutral_mu) if accommodation_layer.neutral_mu is not None else None
		self.neutral_sigma = torch.clone(accommodation_layer.neutral_sigma) if accommodation_layer.neutral_sigma is not None else None

	def potents(self):
		num_classes = self.positive_mu.shape[0]
		num_potents_per_functional_facet = self.positive_mu.shape[1]
		classes = flat_map([[potent] * num_potents_per_functional_facet for potent in range(num_classes)])
		neutral_concepts = None
		if self.neutral_mu is not None:
			neutral_concepts = flat_map([[potent] * num_potents_per_functional_facet for potent in range(self.neutral_mu.shape[0])])
		positive_potents = [Potent(id_=i,
								   means=self.positive_mu[classes[i], i % num_potents_per_functional_facet, :].tolist(),
								   stds=F.softplus(self.positive_sigma[classes[i], i % num_potents_per_functional_facet, :]).tolist(),
								   plasticity=float(self.positive_plasticity[classes[i], i % num_potents_per_functional_facet].float()) if self.positive_plasticity is not None else None,
								   type=PotentType.Positive,
								   clazz=classes[i]) for i in range(self.positive_mu.shape[0] * num_potents_per_functional_facet)]
		negative_potents = [Potent(id_=i + len(positive_potents),
								   means=self.negative_mu[classes[i], i % num_potents_per_functional_facet, :].tolist() if self.negative_mu is not None else None,
								   stds=F.softplus(self.negative_sigma[classes[i], i % num_potents_per_functional_facet, :]).tolist() if self.negative_sigma is not None else None,
								   plasticity=float(self.negative_plasticity[classes[i], i % num_potents_per_functional_facet].float()) if self.negative_plasticity is not None else None,
								   type=PotentType.Negative,
								   clazz=classes[i]) for i in range(self.negative_mu.shape[0] * self.negative_mu.shape[1])] if self.negative_mu is not None else []
		neutral_potents = [Potent(id_=i + len(positive_potents) + len(negative_potents),
								  means=self.neutral_mu[neutral_concepts[i], i % num_potents_per_functional_facet, :].tolist() if self.neutral_mu is not None else None,
								  stds=F.softplus(self.neutral_sigma[neutral_concepts[i], i % num_potents_per_functional_facet, :]).tolist() if self.neutral_sigma is not None else None,
								  plasticity=None,
								  type=PotentType.Neutral,
								  clazz=None) for i in range(self.neutral_mu.shape[0] * self.neutral_mu.shape[1])] if self.neutral_mu is not None else []
		return [*positive_potents, *negative_potents, *neutral_potents]

	def num_potents(self):
		num_potents = self.positive_mu.shape[0] * self.positive_mu.shape[1]
		if self.neutral_mu is not None: num_potents += self.neutral_mu.shape[0] * self.neutral_mu.shape[1]
		if self.negative_mu is not None: num_potents += self.negative_mu.shape[0] * self.negative_mu.shape[1]
		return num_potents

	def num_classes(self):
		return self.positive_mu.shape[0]

