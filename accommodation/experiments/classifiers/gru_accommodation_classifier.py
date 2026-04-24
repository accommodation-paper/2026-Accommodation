from torch import nn

from accommodation.model.accommodation_layer import AccommodationLayer


def instantiate_gru_accommodation_classifier(args, vocab_size):
    return GRUAccommodationClassifier(vocab_size=vocab_size, in_features=args['embedding-dim'], num_classes=args['num-classes'], latent_dim=args['latent-dim'], negative_potents=args['negative-potents'], neutral_potents=args['neutral-potents'], plasticity=args["plasticity"])


class GRUAccommodationClassifier(nn.Module):
    def __init__(self, vocab_size, in_features, num_classes, latent_dim, negative_potents=True, neutral_potents=1, plasticity=True):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.gru = nn.GRU(128, 256, batch_first=True, bidirectional=True)

        self.proj = nn.Linear(512, 128)
        self.accommodation_layer = AccommodationLayer(in_features, num_classes=num_classes, latent_dim=latent_dim, negative_potents=negative_potents, neutral_potents=neutral_potents, plasticity=plasticity)

    def encode(self, x, mask):
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        pooled = (out * mask.unsqueeze(-1)).mean(dim=1)
        return self.proj(pooled)

    def forward(self, x, mask):
        z = self.encode(x, mask)
        return self.accommodation_layer(z)