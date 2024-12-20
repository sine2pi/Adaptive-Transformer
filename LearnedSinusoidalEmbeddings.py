
class LearnedSinusoidalEmbeddings(nn.Module):
    def __init__(self, n_ctx, n_state, checkpointing=False):
        super().__init__()
        self.n_ctx = n_ctx
        self.n_state = n_state
        self.checkpointing = checkpointing

        position = torch.arange(0, n_ctx, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_state, 2).float() * -(math.log(10000.0) / n_state))
        features = torch.zeros(n_ctx, n_state)
        features[:, 0::2] = torch.sin(position * div_term)
        features[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('sinusoidal_features', features)

        self.positional_embeddings = nn.Parameter(self.sinusoidal_features.clone())

    def forward(self, positions):
        if self.checkpointing:
            position_embeddings = checkpoint(lambda x: self.positional_embeddings[x], positions)
        else:
            position_embeddings = self.positional_embeddings[positions]

        position_embeddings = torch.nn.functional.normalize(position_embeddings, p=2, dim=-1)
        return position_embeddings
