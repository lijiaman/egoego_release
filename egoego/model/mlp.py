import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh', is_dropout=False):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for idx, nh in enumerate(hidden_dims):
            self.affine_layers.append(nn.Linear(last_dim, nh))  
            if idx == 0 and is_dropout:
                self.affine_layers.append(nn.Dropout(p=0.5))
            last_dim = nh

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x
