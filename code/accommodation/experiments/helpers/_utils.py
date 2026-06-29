import json
import os

from accommodation.model.accommodation_layer import Pool


def result_path(results_dir: str, configuration: dict, seed: int) -> str:
	policy = configuration["computation"]["policy"]
	neutral = configuration["field"]["neutral_potents"]
	negpot = configuration["field"]["negative_potents"]
	pospot = configuration["field"]["positive_potents"]
	filename = "trace.json"
	return os.path.join(results_dir + policy + f"/POS{pospot}-NEU{neutral}-NEG{negpot}/{seed}", filename)

def save(out_path: str, record: dict) -> None:
		os.makedirs(os.path.dirname(out_path), exist_ok=True)
		with open(out_path, "w") as f:
			json.dump(record, f, indent=2)

def create_configuration(embedding_dim, policy, neutral_potents, num_potents_per_class, latent_dim, negative_potents, num_classes):
		return {
				"field":
					{
						"embedding_dim": embedding_dim,
						"latent_dim": latent_dim,
						"neutral_potents": neutral_potents * num_potents_per_class,
						"positive_potents": num_classes * num_potents_per_class,
						"negative_potents": num_classes * num_potents_per_class if negative_potents else 0
					},
				"computation": {
					"policy": str(policy.name),
					"pooling": str(Pool.MaxMean.name)
			}
		}
