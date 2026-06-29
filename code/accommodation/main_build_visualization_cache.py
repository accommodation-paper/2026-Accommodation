from pathlib import Path

from accommodation.dataset_visualization_builder import build_visualization_cache


RESULTS_DIR = Path("results")
CACHE_DIR = Path("visualization_cache")
DATA_DIR = Path("data")
DATASETS = ["MNIST", "FMNIST", "CIFAR10", "AGNEWS", "IMDB", "YELP"]
TYPES = ("accommodation", "frozen-backbone")
INCLUDE_RICH_CACHE = True
DEVICE = None


def main():
	all_written = []
	for dataset in DATASETS:
		written = build_visualization_cache(
			results_dir=RESULTS_DIR,
			cache_dir=CACHE_DIR,
			data_dir=DATA_DIR,
			dataset=dataset,
			types=TYPES,
			include_rich_cache=INCLUDE_RICH_CACHE,
			device=DEVICE,
		)
		print(f"[{dataset}] built {len(written)} cache file(s)")
		all_written.extend(written)

	if not all_written:
		print("No cache files were generated.")
		return

	print("Generated cache files:")
	for path in all_written:
		print(path)


if __name__ == "__main__":
	main()
