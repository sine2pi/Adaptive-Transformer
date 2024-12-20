
class BiasedCrossAttention(nn.Module):
    def __init__(self, n_state, n_head, dropout_rate=0.001):
        super().__init__()
        self.n_head = n_head
        self.n_state = n_state
        self.head_dim = n_state // n_head

        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

        self.bias = nn.Parameter(torch.zeros(n_head, 1, self.head_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = LayerNorm(n_state)
        
    def forward(self, q, k, v, mask=None):
        batch_size, seq_length, _ = q.size()

        q = self.query(q).view(batch_size, seq_length, self.n_head, self.head_dim)
        k = self.key(k).view(batch_size, seq_length, self.n_head, self.head_dim)
        v = self.value(v).view(batch_size, seq_length, self.n_head, self.head_dim)

        qk = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5) + self.bias
        if mask is not None:
            qk = qk.masked_fill(mask == 0, float('-inf'))

        w = F.softmax(qk, dim=-1)
        w = self.dropout(w)

        out = (w @ v).transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        out = self.norm(self.out(out) + q.view(batch_size, seq_length, -1))
        return out
