import torch.nn as nn
import torch

class CrossLayer(nn.Module):
    def __init__(self, input_dim):
        super(CrossLayer,self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
    def forward(self, x0, x):
        cross = torch.matmul(torch.outer(self.weight, x.squeeze()), x0)
        return cross + self.bias + x

class DCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_cross_layer):
        super(DCN, self).__init__()
        self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(num_cross_layer)])
        self.deep_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),           
            nn.Linear(hidden_dim, output_dim)
        )
        self.fc_out = nn.Linear(input_dim+output_dim, 1)
    def forward(self, x):
        x0 = x
        for layer in self.cross_layers:
            x = layer(x0, x)
        deep_output = self.deep_layer(x)
        concat_output = torch.concat([x, deep_output], dim=1)
        output = self.fc_out(concat_output)
        return output