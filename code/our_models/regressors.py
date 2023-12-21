import torch
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self, dims: list):
        super(Regressor, self).__init__()
        
        [in_features, hidden_features, out_features] = dims
        
        h_neurons = [in_features] + hidden_features
        module_layers = []
        for i in range(len(h_neurons)-1):
            module_layers.append(nn.Linear(h_neurons[i], h_neurons[i+1]))
            module_layers.append(nn.ReLU())
            module_layers.append(nn.BatchNorm1d(h_neurons[i+1]))
            module_layers.append(nn.Dropout(0.2))
        
        self.h_layers = nn.Sequential(*module_layers)
        self.out_layer = nn.Linear(h_neurons[-1], out_features)
        self.out_activation = nn.ReLU()
        
    def forward(self, x):
        x = self.h_layers(x)
        return self.out_activation(self.out_layer(x))





