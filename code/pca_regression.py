# %%
import json
import sys
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from os import path
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
sys.path.append('../code')
from data_loader import Archs4GeneExpressionDataset, GtexDataset
from our_models import Regressor
from torch.utils.data import WeightedRandomSampler, DataLoader

use_cuda = True
TRAIN_EPOCHS = 100 
LEARNING_RATE = 1e-5

n_components = 256
reg_hidden_layers = [2048, 2048]
out_features = 156958

# %%
dat_dir = "data/hdf5"
archsDset = Archs4GeneExpressionDataset(data_dir = dat_dir, load_in_mem=False)
unlabeled_dataloader = DataLoader(archsDset, batch_size=400, num_workers=2, prefetch_factor=1)

gtexDset_train = GtexDataset(data_dir=dat_dir, split="train", load_in_mem=False)
sampler_train = WeightedRandomSampler(weights=gtexDset_train.sample_weights, num_samples=len(gtexDset_train), replacement=True)
training_dataloader = DataLoader(gtexDset_train, sampler=sampler_train, batch_size=64, num_workers=2, prefetch_factor=1)

gtexDset_val = GtexDataset(data_dir=dat_dir, split="val", load_in_mem=False)
sampler_val = WeightedRandomSampler(weights=gtexDset_val.sample_weights, num_samples=len(gtexDset_val), replacement=True)
validation_dataloader = DataLoader(gtexDset_val, batch_size=64, num_workers=2, prefetch_factor=1)


# %%
ipca = IncrementalPCA(n_components=n_components)
pca_model_path = f"pca_model_{n_components}.pkl"

if path.exists(pca_model_path):
    with open(pca_model_path, "rb") as f:
        ipca = pickle.load(f)
else:
    # create PCA incrementally
    for sample in tqdm(unlabeled_dataloader, desc="Fitting PCA"):
        ipca.partial_fit(sample)
    with open(pca_model_path, "wb") as f:
        pickle.dump(ipca, f)

# %%
model = Regressor([n_components, reg_hidden_layers, out_features])
criterion = torch.nn.MSELoss()

if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()

# %%
val_losses, val_iter = [], []
train_losses, train_iter = [], []
val_batches = 0
batches = 0
for epoch in range(TRAIN_EPOCHS):
    model.train()
    for X_train, y_train in training_dataloader:
        train_losses.append([])
        train_iter.append([])
        X_train = ipca.transform(X_train)
        X_train = torch.tensor(X_train, dtype=torch.float32, device='cuda' if use_cuda else 'cpu')
        
        y_pred = model.forward(X_train)
        
        if use_cuda: y_train = y_train.cuda()
        train_loss = criterion(y_pred, y_train)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        train_losses[-1].append(get_numpy(train_loss))
        train_iter[-1].append(batches)
        batches += 1
    
    model.eval()
    for X, y in validation_dataloader:
        val_losses.append([])
        val_iter.append([])
        X = ipca.transform(X)
        X = torch.tensor(X, dtype=torch.float32, device='cuda' if use_cuda else 'cpu')
        if use_cuda: y = y.cuda()
        
        y_pred = model.forward(X)
        
        val_loss = criterion(y_pred, y)
        val_losses[-1].append(get_numpy(val_loss))

    val_iter.append(batches)
    
    print(f"# Epoch {epoch+1}/{TRAIN_EPOCHS}")
    print(f"Training loss:\t{np.mean(train_losses[-1]):4.4f}\tValidation loss:\t{np.mean(val_losses[-1]):4.4f}\n")

with open("regressor_model.pkl", "wb") as f:
    pickle.dump(model, f)
