from torch import nn


def instantiate_mlp_linear_classifier(args):
    return MLPLinearClassifier(input_dim=args['input-dim'], num_classes=args['num-classes'])


class MLPLinearClassifier(nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes=2
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

        self.classification_layer = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classification_layer(self.model(x))
