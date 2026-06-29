from torchvision import datasets
from torch.utils.data import Dataset
from typing import Optional, Callable


class Cifar10Dataset(Dataset):
	def __init__(self,
	             root='./data',
				 train: Optional[bool] = True,
				 transform: Optional[Callable] = None):
		self.dataset = datasets.CIFAR10(
			root=root,
			train=train,
			download=True,
			transform=transform
		)
		self.len = len(self.dataset)

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		image, label = self.dataset[idx]
		return image, label
