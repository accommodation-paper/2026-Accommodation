from torch import nn


def instantiate_mnist_linear_classifier(args):
    return MNISTLinearClassifier(embedding_dim=args['embedding-dim'], num_classes=args['num-classes'])


class MNISTLinearClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_classes: int = 10,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )


        self.classifier = nn.Linear(
            embedding_dim,
			num_classes
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)
