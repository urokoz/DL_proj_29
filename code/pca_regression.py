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


# %%
dat_dir = sys.argv[1]
archsDset = Archs4GeneExpressionDataset(data_dir = dat_dir, load_in_mem=False)
unlabeled_dataloader = DataLoader(archsDset, batch_size=400, num_workers=2, prefetch_factor=1)

gtexDset_train = GtexDataset(data_dir=dat_dir, split="train", load_in_mem=False)
sampler_train = WeightedRandomSampler(weights=gtexDset_train.sample_weights, num_samples=len(gtexDset_train), replacement=True)
training_dataloader = DataLoader(gtexDset_train, sampler=sampler_train, batch_size=64, num_workers=2, prefetch_factor=1)

gtexDset_val = GtexDataset(data_dir=dat_dir, split="val", load_in_mem=False)
sampler_val = WeightedRandomSampler(weights=gtexDset_val.sample_weights, num_samples=len(gtexDset_val), replacement=True)
validation_dataloader = DataLoader(gtexDset_val, batch_size=64, num_workers=2, prefetch_factor=1)


# %%
n_components = 100
ipca = IncrementalPCA(n_components=n_components)
pca_model_path = "pca_model.pkl"

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
use_cuda = True
# out_features = 10000
reg_hidden_layers = [2048, 2048]
out_features = 156958

model = Regressor([n_components, reg_hidden_layers, out_features])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = torch.nn.MSELoss(reduction="sum")

if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()

# %%
epochs = 5
log_every = 5
val_every = 5
tot_batches = 0

val_losses, val_iter = [], []
train_losses, train_iter = [], []

for i in range(epochs):
    tot_train_loss, batches = 0, 0
    for X_train, y_train in training_dataloader:
        if batches % val_every == 0:
            model.eval()
            tot_val_loss = 0
            for X, y in validation_dataloader:
                
                X_encoded = ipca.transform(X)
                
                X_encoded = torch.tensor(X_encoded, dtype=torch.float32, device='cuda' if use_cuda else 'cpu')
                if use_cuda:
                    y = y.cuda()
                
                y_pred = model.forward(X_encoded)
                
                val_loss = criterion(y_pred, y[:, :out_features])
                tot_val_loss += get_numpy(val_loss)

            val_iter.append(batches)
            val_losses.append(tot_val_loss/len(validation_dataloader))

        model.train()
        
        X_encoded = ipca.transform(X_train)
        
        X_encoded = torch.tensor(X_encoded, dtype=torch.float32, device='cuda' if use_cuda else 'cpu')
        
        y_train = y_train[:, :out_features]
        if use_cuda:
            y_train = y_train.cuda()
        
        
        y_pred = model.forward(X_encoded)
        
        train_loss = criterion(y_pred, y_train)
        tot_train_loss += get_numpy(train_loss)
        tot_batches += 1
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
    
        if batches % log_every == 0:
            train_iter.append(batches)
            train_losses.append(tot_train_loss/tot_batches)
            tot_train_loss = 0
            tot_batches = 0
            print(f"# Epoch {i+1}/{epochs}\n# Batch {batches+1}/{len(training_dataloader)}")
            print(f"Training loss:\t{train_losses[-1]}\tValidation loss:\t{val_losses[-1]}")
            # fig = plt.figure(figsize=(12,4))
            # plt.subplot(1, 2, 1)
            # plt.plot(train_iter, train_losses, label='train_loss')
            # plt.plot(val_iter, val_losses, label='valid_loss')
            # plt.legend()
            # plt.show()
    
        batches += 1

with open("regressor_model.pkl", "wb") as f:
    pickle.dump(model, f)
