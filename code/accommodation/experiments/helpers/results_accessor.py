import json
import os


def result_path(type: str, args: dict, seed: int, cycle: int) -> str:
    policy = "Likelihood"
    neutral = args["neutral-potents"]
    negpot = args["negative-potents"]
    filename = (
        f"policy-{policy}"
        f"_neutral-{neutral}"
        f"_negative_potents-{negpot}"
        f"_seed-{seed}"
        f"_cycle-{cycle:02d}.json"
    )
    return os.path.join(args['results-dir'], type, args['dataset'], filename)

def save(out_path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)