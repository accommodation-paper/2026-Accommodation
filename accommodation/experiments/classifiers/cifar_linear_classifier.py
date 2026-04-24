from torch import nn

def instantiate_cifar_accommodation_classifier(args):
    return Cifar10LinearClassifier(embedding_dim=args['embedding-dim'], num_classes=args['num-classes'])


class Cifar10LinearClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        num_classes: int = 10,
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Linear(
			embedding_dim,
			num_classes
		)

    def forward(self, x):
        h = self.backbone(x)
        return self.classifier(h)
