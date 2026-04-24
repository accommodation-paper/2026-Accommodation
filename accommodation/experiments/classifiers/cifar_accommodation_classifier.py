from torch import nn

from accommodation.model.accommodation_layer import AccommodationLayer, Policy, Pool


def instantiate_cifar_accommodation_classifier(args):
    return Cifar10AccommodationClassifier(neutral_potents=args['neutral-potents'], negative_potents=args['negative-potents'], num_classes=args['num-classes'], latent_dim=args['latent-dim'], plasticity=args["plasticity"])


class Cifar10AccommodationClassifier(nn.Module):
    def __init__(
        self,
        policy=Policy.Likelihood,
        neutral_potents: int = 5,
        num_potents_per_class: int = 5,
        latent_dim: int = 30,
        negative_potents: bool = True,
        pool=Pool.MaxMean,
        plasticity: bool = True,
        num_classes: int = 10,
        embedding_dim: int = 256,
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

        self.accommodation_layer = AccommodationLayer(
            embedding_dim,
            policy=policy,
            neutral_potents=neutral_potents,
            num_potents_per_class=num_potents_per_class,
            latent_dim=latent_dim,
            negative_potents=negative_potents,
            pool=pool,
            num_classes=num_classes,
            plasticity=plasticity
        )

    def forward(self, x):
        h = self.backbone(x)
        return self.accommodation_layer(h)
