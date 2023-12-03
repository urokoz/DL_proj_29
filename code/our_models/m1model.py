import torch
import torch.nn as nn
import sys
sys.path.append('code')
sys.path.append("code/wohlert")
from our_models.regressors import Regressor
from models import VariationalAutoencoder


class M1_model(nn.Module):
    def __init__(self, VAE: VariationalAutoencoder, regressor: Regressor):
        super(M1_model, self).__init__()
        
        self.regressor = regressor
        
        # Make vae feature model untrainable by freezing parameters
        self.VAE = VAE
        self.VAE.train(False)

        for param in self.VAE.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        x_sample, _, _ = self.VAE.encoder(x)
               
        return self.regressor.forward(x_sample)

