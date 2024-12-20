
class HybridAttention(nn.Module):
    def __init__(self, n_state, n_head, window_size=1, dropout_rate=0.001):
        super().__init__()
        self.local_attn = nn.MultiheadAttention(n_state, n_head, dropout=dropout_rate)
        self.global_attn = nn.MultiheadAttention(n_state, n_head, dropout=dropout_rate)
        self.ln_local = LayerNorm(n_state)
        self.ln_global = LayerNorm(n_state)

        self.dropout = nn.Dropout(dropout_rate)
        self.window_size = window_size

    def forward(self, x):
        x_local = self.ln_local(x)
        x_global = self.ln_global(x)
        x_local = x_local.permute(1, 0, 2)
        x_global = x_global.permute(1, 0, 2)
        local_out = self.sliding_window_attention(x_local)
        global_out, _ = self.global_attn(x_global, x_global, x_global)
        combined_out = local_out + global_out
        combined_out = combined_out.permute(1, 0, 2)
        return self.dropout(combined_out)

    def sliding_window_attention(self, x):
        batch_size, seq_len, n_state = x.size()
        window_size = min(self.window_size, max(1, seq_len // 4))
        output = torch.zeros_like(x, device=x.device, dtype=x.dtype)

        for i in range(0, seq_len, window_size):
            end = min(i + window_size, seq_len)
            query = x[i:end, :, :]
            start = max(0, i - window_size)
            key = x[start:end, :, :]
            value = x[start:end, :, :]
            attn_output, _ = self.local_attn(query, key, value)
            output[i:end, :, :] = attn_output[:end - i, :, :]

        return output
