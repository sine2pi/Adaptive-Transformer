
class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, max_rel_dist = 1, cross_attention=True, checkpointing=False, base=10000):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = LearnedSinusoidalEmbeddings(n_ctx, n_state, checkpointing=checkpointing)
        self.checkpointing = checkpointing
        self.h_dim = n_state // n_head

        self.combined_rotary = CombinedRotaryEmbedding(
            n_state=n_state,
            n_head=n_head,
            num_rotations=self.h_dim // 2,
            base=base,
            checkpointing=False 
        )

        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, max_rel_dist, checkpointing=checkpointing) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def update_base(self, new_base):
        self.combined_rotary.update_base(new_base)
        for block in self.blocks:
            if isinstance(block.attn, MultiHeadAttention, CombinedRotaryEmbedding):
                block.attn.update_base(new_base)
            if block.cross_attn and isinstance(block.cross_attn, MultiHeadAttention, CombinedRotaryEmbedding):
                block.cross_attn.update_base(new_base)

    def forward(self, x):
        if self.checkpointing:
            x = checkpoint(self._conv_forward, x)
        else:
            x = self._conv_forward(x)

        for block in self.blocks:
            if self.checkpointing:
                x = checkpoint(block, x)
            else:
                x = block(x)

        x = self.ln_post(x)
        return x

    def _conv_forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        x = self.combined_rotary(x)

        pos_emb = self.positional_embedding(torch.arange(x.size(1), device=x.device)).unsqueeze(0)
        x = x + pos_emb
        return x
