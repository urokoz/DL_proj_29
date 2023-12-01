import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from os import path
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
sys.path.append('../code')
sys.path.append("wohlert")
from data_loader import Archs4GeneExpressionDataset, GtexDataset
from models.regressor_model import Regressor
from torch.utils.data import WeightedRandomSampler, DataLoader


# %%
dat_dir = sys.argv[1]
archsDset = Archs4GeneExpressionDataset(data_dir = dat_dir, load_in_mem=False)
unlabeled_dataloader = DataLoader(archsDset, batch_size=400, num_workers=2, prefetch_factor=1)


# NN structure
input_dim         = 18965 # gtex-gene features (can be reduced/capped for quicker tests if needed)
latent_dim        = 32
hidden_layers     = [512]

# Data loader
BATCH_SIZE        = 32
NUM_WORKERS       = 4
PREFETCH_FACTOR   = 2
MAX_FEATURE_VALUE = 1 # Max value of features for normalization

# Training pars
TRAIN_EPOCHS      = 5
LEARNING_RATE     = 1e-4
BETA              = 0.1
ELBO_GAIN         = 1 

# %%
from models import VariationalAutoencoder
from layers import GaussianSample
from inference import log_gaussian, log_standard_gaussian

model = VariationalAutoencoder([input_dim, latent_dim, hidden_layers])
model

