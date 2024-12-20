
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate = 0.001, use_batchnorm: bool = True, activation: str = 'relu'):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.use_batchnorm = use_batchnorm
        self.activation = activation

        if self.use_batchnorm:
            self.batchnorm = nn.BatchNorm1d(out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity=self.activation)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(-1, x.size(-1))  
        x = self.linear(x)

        if self.use_batchnorm:
            x = self.batchnorm(x)

        x = self.apply_activation(x)
        x = self.dropout(x)
        x = x.view(batch_size, seq_len, -1)  
        
        return x

    def apply_activation(self, x):
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        else:
            raise ValueError(f'Unsupported activation function: {self.activation}')
