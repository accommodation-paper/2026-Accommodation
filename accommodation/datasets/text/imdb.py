import os
import torch
import pandas as pd
from accommodation.datasets.text.build_vocab import build_vocab
from torch.utils.data import Dataset


class IMDBDataset(Dataset):
	def __init__(self, datasets_path="data", vocab=None, max_len=256):
		csv_path = os.path.join(datasets_path, "IMDB_dataset.csv")
		self.df = pd.read_csv(csv_path)
		self.texts = self.df['review'].astype(str).tolist()
		self.labels = [1 if s.lower() == 'positive' else 0 for s in self.df['sentiment']]
		self.samples = list(zip(self.texts, self.labels))
		self.max_len = max_len
		if vocab is None:
			self.vocab = build_vocab(self.texts)
		else:
			self.vocab = vocab

		self.vocab_size = len(self.vocab)

	def encode_sentence(self, sentence):
		tokens = sentence.split()
		ids = [self.vocab.get(t, self.vocab['<unk>']) for t in tokens]
		if len(ids) < self.max_len:
			ids += [self.vocab['<pad>']] * (self.max_len - len(ids))
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
