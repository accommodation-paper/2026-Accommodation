from torch import Tensor
from torchvision import datasets
from torch.utils.data import Dataset
from typing import Optional, Callable


class FashionMNISTDataset(Dataset):
	def __init__(self,
	             root: str = "data",
				 train: Optional[bool] = True,
				 transform: Optional[Callable] = None):
		self.dataset = datasets.FashionMNIST(
			root=root,
			train=train,
			download=True,
			transform=transform
		)
		self.len = len(self.dataset)

	def __len__(self):
		return self.len

	def __getitem__(self, idx) -> Tensor:
		image, label = self.dataset[idx]
		return image, label