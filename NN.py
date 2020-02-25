import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):

    def __init__(self, n, D_in, H, D_out, init_weight=True):
        super(NN, self).__init__()
        if n == 1:
            self.layers = nn.ModuleList([nn.Linear(D_in, D_out)])
        else:
            self.layers = nn.ModuleList([nn.Linear(D_in, H)])
            self.layers.extend([nn.Linear(H, H) for _ in range(1, n - 1)])
            self.layers.append(nn.Linear(H, D_out))
        self.activation = nn.ReLU()
        if init_weight is True:
            self.init_weights()

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

    def init_weights(self):
        for layer in self.layers:
            torch.nn.init.uniform_(layer.weight)
            layer.bias.data.fill_(0.01)