# %%
import os
import sys
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from os import path
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA, PCA
sys.path.append('../code')
sys.path.append("code/wohlert")
from data_loader import Archs4GeneExpressionDataset
from torch.utils.data import DataLoader


def generate_data_and_plot(model, latent_dim =256, n_samples=128, use_cuda=True):    
    use_cuda = use_cuda and torch.cuda.is_available()
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

    return generated_data, fig


def trainer(BETA, train_dataloader, val_dataloader, lr=1e-5, TRAIN_EPOCHS=20, elbo_gain=1, use_cuda=True):
    use_cuda = use_cuda and torch.cuda.is_available()
    
    # NN structure
    input_dim         = 18965 # gtex-gene features (can be reduced/capped for quicker tests if needed)
    latent_dim        = 256
    hidden_layers     = [1024, 512]

    #### Model setup ####
    from wohlert.models import VariationalAutoencoder
    from wohlert.layers import GaussianSample
    from wohlert.inference import log_gaussian, log_standard_gaussian


    print(f"{use_cuda=}")
    model = VariationalAutoencoder([input_dim, latent_dim, hidden_layers])
    print(model)

    criterion = torch.nn.MSELoss()

    if use_cuda:
        criterion = criterion.cuda()
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


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
        batch_counter = 0
        
        # TRAINING LOOP
        for u in train_dataloader:
            if torch.max(u) > max_feature:
                max_feature = torch.max(u)
                
            if use_cuda: u = u.cuda()

            reconstruction = model(u)
            
            MSE_batch = elbo_gain * criterion(reconstruction, u)
            KLdiv_batch = elbo_gain * torch.mean(model.kl_divergence)

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
            ELBO_batch = (MSE_batch + BETA * KLdiv_batch)
                    
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

                if use_cuda: u = u.cuda()

                # Get latent variables
                z, _, _ = model.encoder.get_latent_vars(u)
                latent_representations.append(z.detach().cpu().numpy())

                reconstruction = model(u)
                MSE_batch = elbo_gain * criterion(reconstruction, u)
                KLdiv_batch = elbo_gain * torch.mean(model.kl_divergence)
                ELBO_batch = (MSE_batch + BETA * KLdiv_batch)            
                
                log_val_MSE_m[epoch, batch_counter]   = MSE_batch
                log_val_KLdiv_m[epoch, batch_counter] = KLdiv_batch.item()
                log_val_ELBO_m[epoch, batch_counter]  = ELBO_batch.item()
                batch_counter += 1        
                
        # PRINT RESULTS per EPOCH            
        val_ELBO_epoch_now  = log_val_ELBO_m[epoch, :].mean(axis=0)            
        val_MSE_epoch_now   = log_val_MSE_m[epoch, :].mean(axis=0)
        val_KLdiv_epoch_now = log_val_KLdiv_m[epoch, :].mean(axis=0)

        print(f"Beta: {BETA}\tEpoch: {epoch}")
        print(f'train ELBO: {train_ELBO_epoch_now:2.2f}\tMSE: {train_MSE_epoch_now:2.2f}\tKLdiv: {train_KLdiv_epoch_now:2.2f}')
        print(f'val   ELBO: {val_ELBO_epoch_now:2.2f}\tMSE: {val_MSE_epoch_now:2.2f}\tKLdiv: {val_KLdiv_epoch_now:2.2f}\n')


    # List of arrays -> single numpy array
    print(f'Max value after normalization: {max_feature:2.2f}')
    if max_feature > 0: print(f'WARNING: Max value after normalization exceeds 1')
    
    return model
    

if __name__ == '__main__':
    
    #### Data loader setup ####
    dat_dir = "data/hdf5"
    archsDset_train = Archs4GeneExpressionDataset(data_dir = dat_dir, split="train", load_in_mem=False)
    train_dataloader = DataLoader(archsDset_train, batch_size=80, num_workers=2, prefetch_factor=1)

    archsDset_val = Archs4GeneExpressionDataset(data_dir = dat_dir, split="val", load_in_mem=False)
    val_dataloader = DataLoader(archsDset_val, batch_size=80, num_workers=2, prefetch_factor=1)
    
    # Incremental PCA
    n_components = 2
    ipca = IncrementalPCA(n_components=n_components)

    # Fit IPCA on training data
    for data in tqdm(train_dataloader, desc="Fitting IPCA"): 
        ipca.partial_fit(data.cpu().numpy())
    
    # Transform original data (in batches)
    def transform_in_batches(dataloader, pca_model):
        transformed_data_list = []
        for data in tqdm(dataloader, desc="Transforming data"):
            transformed_batch = pca_model.transform(data.cpu().numpy())
            transformed_data_list.append(transformed_batch)
        return np.concatenate(transformed_data_list, axis=0)
    
    original_data_transformed = transform_in_batches(train_dataloader, ipca)
    
    
    betas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    for beta in betas:
        result_path = f"results/beta_{beta}"
        os.makedirs(result_path, exist_ok=True)
        model = trainer(BETA=beta, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
        
        generated_data, fig = generate_data_and_plot(model)
        fig.savefig(f"{result_path}/generated_data.png")
        
        # Transform generated data
        generated_data_transformed = ipca.transform(generated_data)
        fig = plt.figure(figsize=(5, 5))

        # Original data
        plt.scatter(original_data_transformed[:, 0], original_data_transformed[:, 1], alpha=0.7, label='Original Data')
        # Generated data
        plt.scatter(generated_data_transformed[:, 0], generated_data_transformed[:, 1], alpha=0.7, label='Generated Data')

        plt.title('PCA Projection of Original and Generated Arch4 Sequences')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        fig.savefig(f"{result_path}/IPCA_orig_vs_generated_data.png")

        ## OUTPUT DATA
        pca_output = PCA(n_components=n_components)
        pca_gen_data = pca_output.fit_transform(generated_data)

        fig = plt.figure(figsize=(5, 5))

        plt.scatter(pca_gen_data[:, 0], pca_gen_data[:, 1], alpha=0.7, label='z')
        plt.title('PCA of generated data')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        fig.savefig(f"{result_path}/PCA_generated_data.png")
        
        
