
class CombinedRotaryEmbedding(nn.Module):
    def __init__(self, n_state, n_head, num_rotations, base=10000, checkpointing=False):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.h_dim = n_state // n_head
        
        self.num_rotations = num_rotations
        self.base = base
        self.checkpointing = checkpointing
        
        self.thetas = nn.Parameter(torch.zeros(num_rotations))
        self.theta_scale = nn.Parameter(torch.ones(1))  
        
        self.rotation_pairs = nn.Parameter(torch.rand(num_rotations, 2) * self.h_dim)
        self.rotation_matrix = nn.Parameter(torch.eye(self.h_dim))
        self.inv_freq = nn.Parameter(1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim)))
    
    def givens_rotation_matrix(self, n_state, i, j, theta):
        G = torch.eye(n_state, device=theta.device)
        G[i, i] = math.cos(theta)
        G[i, j] = -math.sin(theta)
        G[j, i] = math.sin(theta)
        G[j, j] = math.cos(theta)
        return G
    
    def update_base(self, new_base):
        self.base = new_base
        self.inv_freq = nn.Parameter(1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim)))
    
    def reset_parameters(self):
        nn.init.orthogonal_(self.rotation_matrix)
        nn.init.zeros_(self.thetas)
    
    def forward(self, x):
        if self.checkpointing:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)
    
    def _forward(self, x):
        if x.dim() not in [3, 4]:
            raise ValueError(f"Expected input tensor to be 3D or 4D, but got {x.dim()}D")
        
        if x.dim() == 3:
            batch_size, seq_len, n_state = x.size()
            x = x.view(batch_size, seq_len, self.n_head, self.h_dim)
        else:
            batch_size, seq_len, n_head, h_dim = x.size()
            if n_head != self.n_head or h_dim != self.h_dim:
                raise ValueError(f"Expected n_head {self.n_head} and h_dim {self.h_dim}, but got n_head {n_head} and h_dim {h_dim}")
        
        x = x.reshape(-1, self.h_dim)
        
        for k in range(self.num_rotations):
            i, j = self.rotation_pairs[k].long()
            theta = self.thetas[k] * self.theta_scale  
            
            G = self.givens_rotation_matrix(self.h_dim, i, j, theta)
            x = torch.matmul(x, G)
        
        x = torch.matmul(x, self.rotation_matrix)
        x = x.view(batch_size, seq_len, self.n_head, self.h_dim)
        
        sinusoid_inp = torch.einsum('i, j -> i j', torch.arange(seq_len, device=x.device), self.inv_freq.to(x.device))
        sin = sinusoid_inp.sin()[None, :, None, :]
        cos = sinusoid_inp.cos()[None, :, None, :]
        
        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        
        x = x.view(batch_size, seq_len, self.n_state)
        
        return x
