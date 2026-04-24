import os
import torch
import pandas as pd
from torch.utils.data import Dataset

from accommodation.datasets.text.build_vocab import build_vocab


class AGNewsDataset(Dataset):
    def __init__(self, datasets_path="data", vocab=None, max_len=256):
        csv_path = os.path.join(datasets_path, "AGNews_dataset.csv")
        self.df = pd.read_csv(csv_path)

        if "text" in self.df.columns:
            texts = self.df["text"].astype(str)
        elif "title" in self.df.columns and "description" in self.df.columns:
            texts = self.df["title"].astype(str) + " " + self.df["description"].astype(str)
        else:
            raise ValueError("CSV must contain 'text' or ('title' and 'description').")

        if "label" not in self.df.columns:
            raise ValueError("CSV must contain a 'label' column.")

        self.texts = texts.tolist()

        raw_labels = self.df["label"].tolist()
        self.labels = [int(l) - 1 if int(l) in (1, 2, 3, 4) else int(l) for l in raw_labels]

        self.samples = list(zip(self.texts, self.labels))
        self.max_len = max_len

        if vocab is None:
            self.vocab = build_vocab(self.texts)
        else:
            self.vocab = vocab

        self.vocab_size = len(self.vocab)

    def encode_sentence(self, sentence: str):
        tokens = sentence.split()
        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]

        if len(ids) < self.max_len:
            ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sentence, label = self.samples[idx]
        x = self.encode_sentence(sentence)
        y = torch.tensor(label, dtype=torch.long)
        return x, y
