import os
import json


def read(file: str):
    with open(file, mode="r", encoding='utf-8') as f:
        return json.load(f)

def runs_in(directory: str):
    print(directory)
    return list(set([file[0:file.index("_seed")] for file in os.listdir(directory) if not file.startswith("baseline")]))

def read_cycles(directory: str, run: str):
    return [read(os.path.join(directory, file)) for file in os.listdir(directory) if file.startswith(run)]