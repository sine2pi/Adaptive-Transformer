class MultiHeadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int, max_rel_dist: int = 1, base: int = 10000):
        super().__init__()
        assert n_state % n_head == 0, "n_state must be divisible by n_head"
        self.n_head = n_head
        self.h_dim = n_state // n_head
        assert self.h_dim % 2 == 0, "Head dimension must be even for rotary embeddings"

        self.positional_scaling = nn.Parameter(torch.ones(1))

        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

        self.max_rel_dist = max_rel_dist
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim))
        self.register_buffer('inv_freq', inv_freq)
        self.rel_pos_bias = nn.Embedding(2 * self.max_rel_dist - 1, self.n_head)
        self.rel_pos_bias.weight.data.fill_(0)

        self.combined_rotary = CombinedRotaryEmbedding(
            n_state=n_state,
            n_head=n_head,
            num_rotations=self.h_dim // 2,
            base=base,
            checkpointing=False 
        )

        if device:
            self.to(device)

    def update_base(self, new_base): 
        self.base = new_base 
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim)) 
        self.register_buffer('inv_freq', inv_freq) 
        self.combined_rotary.update_base(new_base)

    def forward(self, x, xa = None, mask = None, kv_cache = None):
        q = self.query(x)

        if kv_cache is None or xa is None or 'k' not in kv_cache:
            k_input = x if xa is None else xa
            k = self.key(k_input)
            v = self.value(k_input)
            if kv_cache is not None:
                kv_cache['k'] = k
                kv_cache['v'] = v
        else:
            k = kv_cache['k']
            v = kv_cache['v']

        q = q.view(q.shape[0], q.shape[1], self.n_head, -1)
        k = k.view(k.shape[0], k.shape[1], self.n_head, -1)
        v = v.view(v.shape[0], v.shape[1], self.n_head, -1)

        q = self.combined_rotary(q) 
        k = self.combined_rotary(k)

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk
    
    def qkv_attention(self, q, k, v, mask = None):
        n_batch, n_ctx, n_state = q.shape

        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        seq_len_q = q.size(2)
        seq_len_k = k.size(2)

        positions = torch.arange(seq_len_q, device=q.device).unsqueeze(1) - torch.arange(seq_len_k, device=q.device).unsqueeze(0)
        positions = positions.clamp(-self.max_rel_dist + 1, self.max_rel_dist - 1) + self.max_rel_dist - 1
        rel_bias = self.rel_pos_bias(positions)  
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)  

        qk = qk + rel_bias

        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        qk = qk.detach()

        return out, qk
