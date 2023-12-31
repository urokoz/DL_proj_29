import torch
import sys
import numpy as np
sys.path.append('code')
from data_loader import GtexDataset
from our_models import Regressor, M1_model
sys.path.append("code/wohlert")
from torch.utils.data import WeightedRandomSampler, DataLoader

vae_path = "results/beta_1e-06/vae_model.pt"

TRAIN_EPOCHS = 200
LEARNING_RATE = 5e-5
MAX_FEATURE_VALUE = 1

latent_dim = 256
reg_hidden_layers = [2048, 2048]
out_features = 156958


dat_dir = "data/hdf5"
gtexDset_train = GtexDataset(data_dir=dat_dir, split="train", normalize=True, load_in_mem=False)
sampler_train = WeightedRandomSampler(weights=gtexDset_train.sample_weights, num_samples=2*len(gtexDset_train), replacement=True)
training_dataloader = DataLoader(gtexDset_train, sampler=sampler_train, batch_size=128, num_workers=2, prefetch_factor=1)

gtexDset_val = GtexDataset(data_dir=dat_dir, split="val", normalize=True, load_in_mem=False)
sampler_val = WeightedRandomSampler(weights=gtexDset_val.sample_weights, num_samples=2*len(gtexDset_val), replacement=True)
validation_dataloader = DataLoader(gtexDset_val, batch_size=128, num_workers=2, prefetch_factor=1)

use_cuda = True and torch.cuda.is_available()

print(f"{use_cuda=}")
vae_model = torch.load(vae_path)

regressor_model = Regressor([latent_dim, reg_hidden_layers, out_features])

model = M1_model(vae_model, regressor_model)
print(model)

criterion = torch.nn.MSELoss()

if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()
    
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()

val_losses, val_iter = [], []
train_losses, train_iter = [], []
val_batches = 0
batches = 0
for epoch in range(TRAIN_EPOCHS):
    model.train()
    for X_train, y_train in training_dataloader:
        train_losses.append([])
        train_iter.append([])
        X_train = X_train / MAX_FEATURE_VALUE
        if use_cuda:
            X_train, y_train = X_train.cuda(), y_train.cuda()
        
        y_pred = model.forward(X_train)
        
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
        X = X / MAX_FEATURE_VALUE
        if use_cuda: X, y = X.cuda(), y.cuda()
        
        y_pred = model.forward(X)
        
        val_loss = criterion(y_pred, y)
        val_losses[-1].append(get_numpy(val_loss))

    val_iter.append(batches)
    
    print(f"Epoch {epoch+1}/{TRAIN_EPOCHS}\tTraining loss:\t{np.mean(train_losses[-1]):4.4f}\tValidation loss:\t{np.mean(val_losses[-1]):4.4f}\n")

# Save model
torch.save(model, 'trained_models/M1_model.pt')

