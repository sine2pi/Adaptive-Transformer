
class DynamicConvAttention(nn.Module):
    def __init__(self, n_state, n_head, kernel_size=3, dropout_rate=0.001):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(n_state, n_state, kernel_size, padding=kernel_size // 2, groups=n_head)
        self.dropout = nn.Dropout(dropout_rate)

        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out_proj = nn.Linear(n_state, n_state)

        self.norm = LayerNorm(n_state)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim != self.n_state:
            raise ValueError(f"Expected embed_dim of {self.n_state}, but got {embed_dim}")

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        x = x.permute(0, 2, 1)
        conv_out = self.conv(x)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.norm(conv_out)
        conv_out = self.dropout(conv_out)

        attention_out = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.n_state ** 0.5), dim=-1)
        attention_out = torch.matmul(attention_out, v)
        
        combined_out = conv_out + attention_out
        combined_out = self.norm(combined_out)
        
        return self.out_proj(self.dropout(combined_out)) + x.permute(0, 2, 1)
