
class TextDecoder(nn.Module):
    def __init__(self, vocab_size, n_ctx, n_state, n_head, n_layer, max_rel_dist = 1, cross_attention=True, checkpointing=False, base=10000):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_state)
        self.positional_embedding = LearnedSinusoidalEmbeddings(n_ctx, n_state, checkpointing=checkpointing)
        self.checkpointing = checkpointing
        self.n_head = n_head
        self.h_dim = n_state // n_head
        
        self.combined_rotary = CombinedRotaryEmbedding(
            n_state=n_state,
            n_head=n_head,
            num_rotations=self.h_dim // 2, 
            base=base,
            checkpointing=False  
        )

        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(n_state, n_head, max_rel_dist, cross_attention, checkpointing=checkpointing)
            for _ in range(n_layer)
        ])
        self.ln = LayerNorm(n_state)
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def update_base(self, new_base):
        self.combined_rotary.update_base(new_base)
        for block in self.blocks:
            if isinstance(block.attn, MultiHeadAttention, CombinedRotaryEmbedding):
                block.attn.update_base(new_base)
            if block.cross_attn and isinstance(block.cross_attn, MultiHeadAttention, CombinedRotaryEmbedding):
                block.cross_attn.update_base(new_base)

    def forward(self, x, xa, kv_cache = None):
        if self.checkpointing:
            x = checkpoint(self._embedding_forward, x, xa, kv_cache)
        else:
            x = self._embedding_forward(x, xa, kv_cache)

        for block in self.blocks:
            if self.checkpointing:
                x = checkpoint(block, x, xa, self.mask, kv_cache)
            else:
                x = block(x, xa, self.mask, kv_cache)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits

    def _embedding_forward(self, x, xa, kv_cache):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        positions = torch.arange(x.shape[1], device=x.device) + offset
        pos_emb = self.positional_embedding(positions).unsqueeze(0)

        x = self.token_embedding(x) + pos_emb
        x = x.to(xa.dtype)

        batch_size, seq_length, embedding_dim = x.shape
        num_heads = self.n_head
        head_dim = embedding_dim // num_heads
        x = x.view(batch_size, seq_length, num_heads, head_dim)

        x = self.combined_rotary(x)

        x = x.view(batch_size, seq_length, embedding_dim)
        return x
