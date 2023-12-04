# %%
import pickle
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
sys.path.append('../code')
sys.path.append("code/wohlert")
from data_loader import Archs4GeneExpressionDataset
from torch.utils.data import DataLoader


# NN structure
input_dim         = 18965 # gtex-gene features (can be reduced/capped for quicker tests if needed)
latent_dim        = 32
hidden_layers     = [512]

# Data loader
BATCH_SIZE        = 80
NUM_WORKERS       = 4
PREFETCH_FACTOR   = 2
MAX_FEATURE_VALUE = 24 # Max value of features for normalization

# Training pars
TRAIN_EPOCHS      = 100
LEARNING_RATE     = 1e-3
BETA              = 10
ELBO_GAIN         = 1 


#### Data loader setup ####
dat_dir = "data/hdf5"
archsDset_train = Archs4GeneExpressionDataset(data_dir = dat_dir, split="train", load_in_mem=False)
train_dataloader = DataLoader(archsDset_train, batch_size=BATCH_SIZE, num_workers=2, prefetch_factor=1)

archsDset_val = Archs4GeneExpressionDataset(data_dir = dat_dir, split="val", load_in_mem=False)
val_dataloader = DataLoader(archsDset_val, batch_size=BATCH_SIZE, num_workers=2, prefetch_factor=1)

#### Model setup ####
from wohlert.models import VariationalAutoencoder
from wohlert.layers import GaussianSample
from wohlert.inference import log_gaussian, log_standard_gaussian

use_cuda = True and torch.cuda.is_available()

print(f"{use_cuda=}")
model = VariationalAutoencoder([input_dim, latent_dim, hidden_layers])
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss(reduction="sum")

if use_cuda:
    criterion = criterion.cuda()
    model = model.cuda()


def generate_data_and_plot(epoch = -1, n_samples=128, use_cuda=True):
    model.eval()
    random_sequence = torch.randn(n_samples, latent_dim)
    random_sequence = random_sequence.cuda() if use_cuda else random_sequence
    t_generated_data = model.sample(random_sequence)

    generated_data = t_generated_data.detach().cpu().numpy()

    sample_idx = 1

    fig, ax = plt.subplots(2, 1, figsize=(9,5))

    ax[0].plot(generated_data[sample_idx,:])
    ax[0].grid('both')
    ax[1].plot(generated_data[sample_idx  ,:], 'r')
    ax[1].plot(generated_data[sample_idx+1,:], 'b')
    ax[1].plot(generated_data[sample_idx+2,:], 'k')
    ax[1].plot(generated_data[sample_idx+3,:], 'g')
    ax[1].plot(generated_data[sample_idx+4,:], 'c')
    ax[1].set_xlim((0,100))
    ax[1].grid('both')

    plt.show(block=False)

    return generated_data


log_train_ELBO_m       = np.zeros([TRAIN_EPOCHS, len(train_dataloader)])
log_train_MSE_m        = np.zeros([TRAIN_EPOCHS, len(train_dataloader)])
log_train_KLdiv_m      = np.zeros([TRAIN_EPOCHS, len(train_dataloader)])
log_val_ELBO_m         = np.zeros([TRAIN_EPOCHS, len(val_dataloader)])
log_val_MSE_m          = np.zeros([TRAIN_EPOCHS, len(val_dataloader)])
log_val_KLdiv_m        = np.zeros([TRAIN_EPOCHS, len(val_dataloader)])

log_manual_KLdiv_m     = np.zeros([TRAIN_EPOCHS, len(train_dataloader)])
log_manual_qz_m        = np.zeros([TRAIN_EPOCHS, len(train_dataloader)])
log_manual_pz_m        = np.zeros([TRAIN_EPOCHS, len(train_dataloader)])
log_manual_z_mu_m      = np.zeros([TRAIN_EPOCHS, len(train_dataloader)])
log_manual_z_log_var_m = np.zeros([TRAIN_EPOCHS, len(train_dataloader)])

# Matrix structure:
#    ----- batches -----> 
#   |
# epochs
#   |
#   v

max_feature = 0

latent_representations = []

for epoch in range(TRAIN_EPOCHS):
    model.train()

    epoch_train_ELBO  = 0
    epoch_train_MSE   = 0
    epoch_train_KLdiv = 0
    epoch_val_ELBO    = 0
    epoch_val_MSE     = 0
    epoch_val_KLdiv   = 0    
    batch_counter     = 0
    
    # TRAINING LOOP
    for u in train_dataloader:
        u = u[:,:input_dim] / MAX_FEATURE_VALUE # quick and dirt normalization
        if torch.max(u) > max_feature:
            max_feature = torch.max(u)
            
        if use_cuda: u = u.cuda()

        reconstruction = model(u)
        
        MSE_batch = ELBO_GAIN * criterion(reconstruction, u)
        KLdiv_batch = ELBO_GAIN * BETA * torch.mean(model.kl_divergence)

        # MANUAL calculation of KL_div:        
        z, z_mu, z_log_var = model.encoder(u)            
        # pz: log prob. of "z" under PRIOR N(0,1). The higher (closer to zero) the more likely "z" is the prior.
        pz = log_standard_gaussian(z)
        # qz: log prob. of "z" under the Gaussian given "x". The higher (closer to zero)) the more likely "z" is to encode "x" according to the model.
        qz = log_gaussian(z, z_mu, z_log_var) 
        kl = qz - pz            
        log_manual_z_mu_m[epoch, batch_counter]      = z_mu.mean().item()
        log_manual_z_log_var_m[epoch, batch_counter] = z_log_var.mean().item()
        log_manual_qz_m[epoch, batch_counter]        = qz.mean().item()
        log_manual_pz_m[epoch, batch_counter]        = pz.mean().item()
        log_manual_KLdiv_m[epoch, batch_counter]     = BETA * kl.mean().item()        

        # ELBO is maximized in a VAE.
        # By inverting ELBO, the terms inside are in practice minimized:
        # minimizing mse_loss -> good sample reconstruction
        # minimizing KL_div   -> "moves" distribution of latent vars towards prior distributions (acting as a regularizer)
        ELBO_batch = (MSE_batch + KLdiv_batch)
                
        ELBO_batch.backward()
        optimizer.step()
        optimizer.zero_grad()
                        
        log_train_MSE_m[epoch, batch_counter]   = MSE_batch
        log_train_KLdiv_m[epoch, batch_counter] = KLdiv_batch.item()
        log_train_ELBO_m[epoch, batch_counter]  = ELBO_batch.item()

        batch_counter += 1        

    train_ELBO_epoch_now  = log_train_ELBO_m[epoch, :].mean(axis=0)
    train_MSE_epoch_now   = log_train_MSE_m[epoch, :].mean(axis=0)
    train_KLdiv_epoch_now = log_train_KLdiv_m[epoch, :].mean(axis=0)

    # VALIDATION LOOP
    batch_counter = 0
    model.eval()
    # _ = generate_data_and_plot(epoch)

    with torch.inference_mode():                
        for u in val_dataloader:
            u = u[:,:input_dim] / MAX_FEATURE_VALUE # quick and dirt normalization

            if use_cuda: u = u.cuda(device=0)

            # Get latent variables
            z, _, _ = model.encoder.get_latent_vars(u)
            latent_representations.append(z.detach().cpu().numpy())

            reconstruction = model(u)
            MSE_batch = ELBO_GAIN * criterion(reconstruction, u)
            KLdiv_batch = ELBO_GAIN * BETA * torch.mean(model.kl_divergence)
            ELBO_batch = (MSE_batch + KLdiv_batch)            
            
            log_val_MSE_m[epoch, batch_counter]   = MSE_batch
            log_val_KLdiv_m[epoch, batch_counter] = KLdiv_batch.item()
            log_val_ELBO_m[epoch, batch_counter]  = ELBO_batch.item()
            batch_counter += 1        
            
    # PRINT RESULTS per EPOCH            
    val_ELBO_epoch_now  = log_val_ELBO_m[epoch, :].mean(axis=0)            
    val_MSE_epoch_now   = log_val_MSE_m[epoch, :].mean(axis=0)
    val_KLdiv_epoch_now = log_val_KLdiv_m[epoch, :].mean(axis=0)

    if (epoch >= 2):
        overfit_warning_str = ''        
        val_ELBO_epoch_old1 = log_val_ELBO_m[epoch-1, :].mean(axis=0)
        val_ELBO_epoch_old2 = log_val_ELBO_m[epoch-2, :].mean(axis=0)

        # Naive overfit detection (where validation goes down and then up)
        if np.min([val_ELBO_epoch_now, val_ELBO_epoch_old1, val_ELBO_epoch_old2]) == val_ELBO_epoch_old1:
            overfit_warning_str = ' <--- Possible overfit'

    print(f'Epoch: {epoch}\ttrain ELBO: {train_ELBO_epoch_now:2.2f}\tMSE: {train_MSE_epoch_now:2.2f}\tKLdiv: {train_KLdiv_epoch_now:2.2f}')
    print(f'                val   ELBO: {val_ELBO_epoch_now:2.2f}\tMSE: {val_MSE_epoch_now:2.2f}\tKLdiv: {val_KLdiv_epoch_now:2.2f}\n')


# List of arrays -> single numpy array
latent_representations = np.concatenate(latent_representations, axis=0)
print(f'Max value after normalization: {max_feature:2.2f}')
if max_feature > 0: print(f'WARNING: Max value after normalization exceeds 1')

# Save model
torch.save(model, 'trained_models/vae_model.pt')
