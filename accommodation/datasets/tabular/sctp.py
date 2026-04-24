from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from typing import Optional
import pandas as pd
import torch
import os


class SantanderCustomerTransactionDataset(Dataset):
	def __init__(
			self,
			indices=None,
			csv_path: str = "data",
			scaler: Optional[StandardScaler] = None,
			fit_scaler: bool = False,
	):
		csv_path = os.path.join(csv_path, "SCTP_dataset.csv")
		df = pd.read_csv(csv_path)

		X = df.drop(columns=["ID_code", "target"]).values
		y = df["target"].values

		if indices is not None:
			X = X[indices]
			y = y[indices]

		if scaler is None:
			self.scaler = StandardScaler()
			X = self.scaler.fit_transform(X)
		else:
			self.scaler = scaler
			if fit_scaler:
				X = self.scaler.fit_transform(X)
			else:
				X = self.scaler.transform(X)

		self.X = torch.tensor(X, dtype=torch.float32)
		self.y = torch.tensor(y, dtype=torch.long)

	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]