from torch import nn

def instantiate_gru_linear_classifier(args, vocab_size):
    return GRUClassifier(vocab_size=vocab_size, embedding_dim=args['embedding-dim'], num_classes=args['num-classes'], hidden_dim=args['hidden-dim'])


class GRUClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim=256,
                 num_layers=2,
                 hidden_dim=512,
                 dropout=0.2,
                 proj=128,
                 bidirectional=True,
                 num_classes=2
                 ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_dim = hidden_dim * (2 if bidirectional else 1)

        self.proj = nn.Sequential(
            nn.Linear(self.output_dim, proj),
            nn.ReLU(),
        )

        self.classification_layer = nn.Linear(proj, num_classes)

    def masked_mean(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def forward(self, input_ids, attention_mask):
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        embeds = self.embedding(input_ids)
        out, _ = self.gru(embeds)
        pooled = self.masked_mean(out, attention_mask)
        z = self.proj(pooled)
        return self.classification_layer(z)
