import torch
import torch.nn as nn
import sys
sys.path.append('code')
from data_loader import GtexDataset
from our_models import Regressor, M1_model
sys.path.append("code/wohlert")
from wohlert.models import VariationalAutoencoder
from torch.utils.data import WeightedRandomSampler, DataLoader


TRAIN_EPOCHS = 5

input_dim = 4096
latent_dim = 32
vae_hidden_layers = [512, 256]
reg_hidden_layers = [2048, 2048]
out_features = 156958

LEARNING_RATE = 1e-4

dat_dir = "data/hdf5"
gtexDset_train = GtexDataset(data_dir=dat_dir, split="train", load_in_mem=False)
sampler_train = WeightedRandomSampler(weights=gtexDset_train.sample_weights, num_samples=len(gtexDset_train), replacement=True)
training_dataloader = DataLoader(gtexDset_train, sampler=sampler_train, batch_size=64, num_workers=2, prefetch_factor=1)

gtexDset_val = GtexDataset(data_dir=dat_dir, split="val", load_in_mem=False)
sampler_val = WeightedRandomSampler(weights=gtexDset_val.sample_weights, num_samples=len(gtexDset_val), replacement=True)
validation_dataloader = DataLoader(gtexDset_val, batch_size=64, num_workers=2, prefetch_factor=1)

use_cuda = True and torch.cuda.is_available()

print(f"{use_cuda=}")
vae_path = "trained_models/vae_model.pt"
vae_model = torch.load(vae_path)

regressor_model = Regressor([latent_dim, reg_hidden_layers, out_features])

model = M1_model(vae_model, regressor_model)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss(reduction="sum")

if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()
    
    
def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()


log_every = 25
val_every = 25
tot_batches = 0

val_losses, val_iter = [], []
train_losses, train_iter = [], []
    
for epoch in range(TRAIN_EPOCHS):
    tot_train_loss, batches = 0, 1
    for X_train, y_train in training_dataloader:
        if batches % val_every == 0:
            model.eval()
            tot_val_loss = 0
            for X, y in validation_dataloader:
                if use_cuda: X, y = X.cuda(), y.cuda()
                
                y_pred = model.forward(X)
                
                val_loss = criterion(y_pred, y[:, :out_features])
                tot_val_loss += get_numpy(val_loss)

            val_iter.append(batches)
            val_losses.append(tot_val_loss/len(validation_dataloader))

        model.train()
        y_train = y_train[:, :out_features]
        if use_cuda:
            X_train, y_train = X_train.cuda(), y_train.cuda()
        
        
        y_pred = model.forward(X_train)
        
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
            print(f"# Epoch {epoch+1}/{TRAIN_EPOCHS}\n# Batch {batches+1}/{len(training_dataloader)}")
            print(f"Training loss:\t{round(train_losses[-1], 2)}\tValidation loss:\t{round(val_losses[-1], 2)}")
            # fig = plt.figure(figsize=(12,4))
            # plt.subplot(1, 2, 1)
            # plt.plot(train_iter, train_losses, label='train_loss')
            # plt.plot(val_iter, val_losses, label='valid_loss')
            # plt.legend()
            # plt.show()
    
        batches += 1

# Save model
torch.save(model, 'trained_models/M1_model.pt')

