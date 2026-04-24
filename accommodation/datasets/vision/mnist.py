from typing import Optional, Callable
from torch.utils.data import Dataset
from torchvision import datasets


class MNISTDataset(Dataset):
    def __init__(self,
                 root: str ="./data",
                 train: bool = True,
                 transform: Optional[Callable] = None):
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transform
        )
        self.len = len(self.dataset)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y
