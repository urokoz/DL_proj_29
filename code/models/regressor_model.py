import torch
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self, dims: list):
        super(Regressor, self).__init__()
        
        self.relu = nn.ReLU()
        self.FFN = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU()
        )
        
    def forward(self, x):
        
        return self.FFN(x)





