import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from accommodation.datasets.text.build_vocab import build_vocab


class YelpReviewsStarsDataset(Dataset):
    def __init__(self, datasets_path="data", vocab=None, max_len=256, task="polarity"):
        csv_path = os.path.join(datasets_path, "YELP_dataset.csv")
        df = pd.read_csv(csv_path)

        if "text" not in df.columns:
            raise ValueError("CSV must contain a 'text' column.")
        if "stars" not in df.columns:
            raise ValueError("CSV must contain a 'stars' column (1..5).")

        df["text"] = df["text"].astype(str)
        df["stars"] = pd.to_numeric(df["stars"], errors="coerce")

        df = df.dropna(subset=["text", "stars"])
        df["stars"] = df["stars"].astype(int)

        if task == "stars":
            labels = (df["stars"] - 1).tolist()
            texts = df["text"].tolist()

        elif task == "polarity":
            df = df[df["stars"].isin([1, 2, 4, 5])].copy()
            texts = df["text"].tolist()
            labels = [0 if s in (1, 2) else 1 for s in df["stars"].tolist()]

        else:
            raise ValueError("task must be 'stars' or 'polarity'")

        self.texts = texts
        self.labels = labels
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
