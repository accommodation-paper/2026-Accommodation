from torch import nn

from accommodation.model.accommodation_layer import AccommodationLayer, Policy, Pool


def instantiate_mlp_accommodation_classifier(args):
    return MLPAccommodation(input_dim=args['input-dim'], neutral_potents=args['neutral-potents'], negative_potents=args['negative-potents'], num_classes=args['num-classes'], latent_dim=args['latent-dim'],  plasticity=args["plasticity"])


class MLPAccommodation(nn.Module):
    def __init__(self,
                 input_dim,
                 policy=Policy.Likelihood,
                 neutral_potents=5,
                 num_potents_per_class=5,
                 latent_dim=30,
                 negative_potents=True,
                 pool=Pool.MaxMean,
                 num_classes=2,
                 plasticity=True,
                 ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.accommodation_layer = AccommodationLayer(
            128,
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
        return self.accommodation_layer(self.model(x))
