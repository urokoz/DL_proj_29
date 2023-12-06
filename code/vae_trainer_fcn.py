import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
import random
import math
import sys
sys.path.append('../code')
sys.path.append("code/wohlert")
from models import VariationalAutoencoder
from layers import GaussianSample
from inference import log_gaussian, log_standard_gaussian
from data_loader import Archs4GeneExpressionDataset
from torch.utils.data import DataLoader
from sklearn.decomposition import IncrementalPCA
import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED=42
def vae_trainer_fcn(BATCH_SIZE, NUM_WORKERS, PREFETCH_FACTOR, TRAIN_EPOCHS, LEARNING_RATE, BETA, HIDDEN_NUM_LAYERS, LATENT_DIM, HIDDEN_LAYERS, experiment_number, exp_file, train_dl, val_dl):
 
  torch.manual_seed(RANDOM_SEED)
  np.random.seed(RANDOM_SEED)
  use_cuda = True and torch.cuda.is_available()
  print(f'cuda: {use_cuda}')

  fig_objects = [] # list of figures to be saved to a file
  fig_names   = []

  # NN structure
  input_dim         = 18965 # gtex-gene features (can be reduced/capped for quicker tests if needed)  
  MAX_FEATURE_VALUE = 1 # Max value of features for normalization

  # Training pars
  ELBO_GAIN         = 1  

  variable_info = [
    ("input_dim",     input_dim),
    ("LATENT_DIM",    LATENT_DIM),
    ("HIDDEN_LAYERS", HIDDEN_LAYERS),     
    ("BATCH_SIZE",    BATCH_SIZE),

    ("NUM_WORKERS",    NUM_WORKERS),
    ("PREFETCH_FACTOR",PREFETCH_FACTOR),
    ("TRAIN_EPOCHS",   TRAIN_EPOCHS),
    ("LEARNING_RATE",  LEARNING_RATE),
    ("BETA",           BETA),
    ("ELBO_GAIN",      ELBO_GAIN), 
  ]

  model = VariationalAutoencoder([input_dim, LATENT_DIM, HIDDEN_LAYERS]) 

  ## OPTIMIZER + CRITERION
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
  criterion = torch.nn.MSELoss()

  if use_cuda:
    criterion = criterion.cuda()
    model = model.cuda()

  ## TRAINING/VALIDATION LOOP
  log_train_ELBO_m       = np.zeros([TRAIN_EPOCHS, len(train_dl)])
  log_train_MSE_m        = np.zeros([TRAIN_EPOCHS, len(train_dl)])
  log_train_KLdiv_m      = np.zeros([TRAIN_EPOCHS, len(train_dl)])
  log_val_ELBO_m         = np.zeros([TRAIN_EPOCHS, len(val_dl)])
  log_val_MSE_m          = np.zeros([TRAIN_EPOCHS, len(val_dl)])
  log_val_KLdiv_m        = np.zeros([TRAIN_EPOCHS, len(val_dl)])

  log_manual_KLdiv_m     = np.zeros([TRAIN_EPOCHS, len(train_dl)])
  log_manual_qz_m        = np.zeros([TRAIN_EPOCHS, len(train_dl)])
  log_manual_pz_m        = np.zeros([TRAIN_EPOCHS, len(train_dl)])
  log_manual_z_mu_m      = np.zeros([TRAIN_EPOCHS, len(train_dl)])
  log_manual_z_log_var_m = np.zeros([TRAIN_EPOCHS, len(train_dl)])

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
    batch_counter     = 0
     
    # TRAINING LOOP
    for u in train_dl:
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
    
    if epoch > 0:
      if np.isnan(train_ELBO_epoch_now) or \
        np.isnan(train_MSE_epoch_now) or \
        np.isnan(train_KLdiv_epoch_now) or \
        np.max([train_ELBO_epoch_now, train_MSE_epoch_now, train_KLdiv_epoch_now]) > 1e6:
        print('NaN or too high values values detected:')
        print(f'train_ELBO_epoch_now: {train_ELBO_epoch_now}')
        print(f'train_MSE_epoch_now: {train_MSE_epoch_now}')
        print(f'train_KLdiv_epoch_now: {train_KLdiv_epoch_now}')            
        print('EXPERIMENT ABORTED.')
        save_results(f'{exp_file}{experiment_number}_ABORTED/parameters_{experiment_number}_ABORTED.txt', fig_objects, fig_names, variable_info, model)
        return

    # VALIDATION LOOP
    batch_counter = 0
    model.eval()
    _, fig_objects, fig_names = generate_data_and_plot(model, LATENT_DIM, fig_objects, fig_names, epoch, 128, use_cuda)                            

    with torch.inference_mode():                
      for u in val_dl:
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

    print(f'Epoch: {epoch}\ttrain ELBO: {train_ELBO_epoch_now:2.5f}\tMSE: {train_MSE_epoch_now:2.5f}\tKLdiv: {train_KLdiv_epoch_now:2.5f}')
    print(f'                val   ELBO: {val_ELBO_epoch_now:2.5f}\tMSE: {val_MSE_epoch_now:2.5f}\tKLdiv: {val_KLdiv_epoch_now:2.5f}\n')


  # List of arrays -> single numpy array
  latent_representations = np.concatenate(latent_representations, axis=0)
  if MAX_FEATURE_VALUE != 1:
    print(f'Max value after normalization: {max_feature:2.2f}')
    if max_feature > 0: print(f'WARNING: Max value after normalization exceeds 1')

  ## TRAINING/VALIDATION PLOTS
  # concatenate recorded batch losses for each epoch
  train_ELBO_vs_all_batches  = log_train_ELBO_m.reshape(TRAIN_EPOCHS*len(train_dl),1).squeeze()
  val_ELBO_vs_all_batches    = log_val_ELBO_m.reshape(TRAIN_EPOCHS*len(val_dl),1).squeeze()
  train_MSE_vs_all_batches   = log_train_MSE_m.reshape(TRAIN_EPOCHS*len(train_dl),1).squeeze()
  val_MSE_vs_all_batches     = log_val_MSE_m.reshape(TRAIN_EPOCHS*len(val_dl),1).squeeze()
  train_KLdiv_vs_all_batches = log_train_KLdiv_m.reshape(TRAIN_EPOCHS*len(train_dl),1).squeeze()
  val_KLdiv_vs_all_batches   = log_val_KLdiv_m.reshape(TRAIN_EPOCHS*len(val_dl),1).squeeze()

  # Poster: For plotting purposes
  train_ELBO_vs_all_batches = [0 if x > 100 else x for x in train_ELBO_vs_all_batches]
  train_MSE_vs_all_batches = [0 if x > 100 else x for x in train_MSE_vs_all_batches]
  train_KLdiv_vs_all_batches = [0 if x > 100 else x for x in train_KLdiv_vs_all_batches]

  batch_ratio   = len(train_ELBO_vs_all_batches) / len(val_ELBO_vs_all_batches)
  xaxis_train_v = np.arange(len(train_MSE_vs_all_batches))
  xaxis_val_v   = np.arange(len(val_MSE_vs_all_batches)) * batch_ratio
  zoom_idx1     = -min(100, len(val_dl)*TRAIN_EPOCHS) # maximum 100 samples
  zoom_idx2     = -1

  # 2x2 grid 
  fig = plt.figure(figsize=(10, 6))
  gs = gridspec.GridSpec(2, 2, width_ratios=[4, 2])
  fig_objects.append(fig)
  fig_names.append("train_val_losses")

  ax1 = plt.subplot(gs[0,0])
  ax1.plot(xaxis_train_v, train_MSE_vs_all_batches, 'b')  
  ax1.plot(xaxis_train_v, train_KLdiv_vs_all_batches, 'g')  
  ax1.plot(xaxis_train_v, train_ELBO_vs_all_batches, 'r')  
  ax1.legend(['MSE', 'KLdiv', 'ELBO'])
  ax1.set_ylabel('TRAIN: batch loss') 
  ax1.set_title(f'               ELBO train: {train_ELBO_vs_all_batches[-1]:2.3f}  val: {val_ELBO_vs_all_batches[-1]:2.3f}\n\
               MSE train: {train_MSE_vs_all_batches[-1]:2.3f}  val: {val_MSE_vs_all_batches[-1]:2.3f}\n\
                KLdiv train: {train_KLdiv_vs_all_batches[-1]:2.3f}  val: {val_KLdiv_vs_all_batches[-1]:2.3f}')

  ax1.grid('both')

  ax2 = plt.subplot(gs[1,0])
  ax2.plot(xaxis_val_v, val_MSE_vs_all_batches, 'b')  
  ax2.plot(xaxis_val_v, val_KLdiv_vs_all_batches, 'g')  
  ax2.plot(xaxis_val_v, val_ELBO_vs_all_batches, 'r')  
  ax2.legend(['MSE', 'KLdiv', 'ELBO'])
  ax2.set_xlabel('batch number')
  ax2.set_ylabel('VAL: batch loss') 
  ax2.grid('both')

  ax3 = plt.subplot(gs[0,1])
  ax3.plot(xaxis_train_v, train_MSE_vs_all_batches, 'b')  
  ax3.plot(xaxis_train_v, train_KLdiv_vs_all_batches, 'g')  
  ax3.plot(xaxis_train_v, train_ELBO_vs_all_batches, 'r')  
  ax3.legend(['MSE', 'KLdiv', 'ELBO'], loc='lower left')
  ax3.set_ylabel('TRAIN: batch loss') 
  ax3.set_xlim(xaxis_train_v[zoom_idx1], xaxis_train_v[zoom_idx2])
  ax3.set_ylim(0, 1.2*np.max(train_ELBO_vs_all_batches[zoom_idx1:zoom_idx2]))
  ax3.grid('both')

  ax4 = plt.subplot(gs[1,1])
  ax4.plot(xaxis_val_v, val_MSE_vs_all_batches, 'b')  
  ax4.plot(xaxis_val_v, val_KLdiv_vs_all_batches, 'g')  
  ax4.plot(xaxis_val_v, val_ELBO_vs_all_batches, 'r')  
  ax4.legend(['MSE', 'KLdiv', 'ELBO'], loc='lower left')
  ax4.set_xlabel('batch number')
  ax4.set_ylabel('VAL: batch loss') 
  ax4.set_xlim(xaxis_val_v[zoom_idx1], xaxis_val_v[zoom_idx2])
  ax4.set_ylim(0, 1.2*np.max(val_ELBO_vs_all_batches[zoom_idx1:zoom_idx2]))
  ax4.grid('both')
  plt.close()

  fig = plt.figure(figsize=(10,3))
  fig_objects.append(fig)
  fig_names.append("train_val_elbo_over_epochs")

  plt.plot(np.mean(log_train_ELBO_m, axis=1), 'r', label='train loss')
  plt.plot(np.mean(log_val_ELBO_m, axis=1), 'b', label='val loss')  
  plt.title('ELBO over epochs')
  plt.xlabel('epoch')
  plt.ylabel('MSE loss')
  plt.grid('both')
  plt.legend()
  plt.close()
  

  ## PLOTS of MANUAL KLdiv calculations

  z_mu_vs_all_batches      = log_manual_z_mu_m.reshape(TRAIN_EPOCHS * len(train_dl), 1).squeeze()
  z_log_var_vs_all_batches = log_manual_z_log_var_m.reshape(TRAIN_EPOCHS * len(train_dl), 1).squeeze()
  qz_vs_all_batches        = log_manual_qz_m.reshape(TRAIN_EPOCHS * len(train_dl), 1).squeeze()
  pz_vs_all_batches        = log_manual_pz_m.reshape(TRAIN_EPOCHS * len(train_dl), 1).squeeze()
  kl_div_vs_all_batches    = log_manual_KLdiv_m.reshape(TRAIN_EPOCHS * len(train_dl), 1).squeeze()

  # 4x2 grid 
  fig = plt.figure(figsize=(10, 8))
  gs = gridspec.GridSpec(5, 2, width_ratios=[4, 2])
  fig_objects.append(fig)
  fig_names.append("kldiv_intermediate_calcs")

  ax_0_0 = plt.subplot(gs[0,0])
  ax_0_0.plot(xaxis_train_v, kl_div_vs_all_batches, 'r')
  ax_0_0.set_title('Mean of KL divergence')
  ax_0_0.grid(True)


  ax_1_0 = plt.subplot(gs[1,0])
  ax_1_0.plot(xaxis_train_v, z_mu_vs_all_batches, 'g')
  ax_1_0.set_title('Mean of z_mu')
  ax_1_0.grid('both')

  ax_2_0 = plt.subplot(gs[2,0])
  ax_2_0.plot(xaxis_train_v, z_log_var_vs_all_batches, 'orange')
  ax_2_0.set_title('Mean of z_log_var')
  ax_2_0.grid('both')

  ax_3_0 = plt.subplot(gs[3,0])
  ax_3_0.plot(xaxis_train_v, qz_vs_all_batches, 'purple')
  ax_3_0.set_title('Mean of qz')
  ax_3_0.grid('both')

  ax_4_0 = plt.subplot(gs[4,0])
  ax_4_0.plot(xaxis_train_v, pz_vs_all_batches, 'yellow')
  ax_4_0.set_title('Mean of pz')
  ax_4_0.grid('both')

  # ZOOM
  ax_0_1 = plt.subplot(gs[0,1])
  ax_0_1.plot(xaxis_train_v[zoom_idx1:zoom_idx2], kl_div_vs_all_batches[zoom_idx1:zoom_idx2], 'r')
  ax_0_1.set_title('Mean of KL divergence')
  ax_0_1.grid('both')

  ax_1_1 = plt.subplot(gs[1,1])
  ax_1_1.plot(xaxis_train_v[zoom_idx1:zoom_idx2], z_mu_vs_all_batches[zoom_idx1:zoom_idx2], 'g')
  ax_1_1.set_title('Mean of z_mu')
  ax_1_1.grid('both')

  ax_2_1 = plt.subplot(gs[2,1])
  ax_2_1.plot(xaxis_train_v[zoom_idx1:zoom_idx2], z_log_var_vs_all_batches[zoom_idx1:zoom_idx2], 'orange')
  ax_2_1.set_title('Mean of z_log_var')
  ax_2_1.grid('both')

  ax_3_1 = plt.subplot(gs[3,1])
  ax_3_1.plot(xaxis_train_v[zoom_idx1:zoom_idx2], qz_vs_all_batches[zoom_idx1:zoom_idx2], 'purple')
  ax_3_1.set_title('Mean of qz')
  ax_3_1.grid('both')

  ax_4_1 = plt.subplot(gs[4,1])
  ax_4_1.plot(xaxis_train_v[zoom_idx1:zoom_idx2], pz_vs_all_batches[zoom_idx1:zoom_idx2], 'yellow')
  ax_4_1.set_title('Mean of pz')
  ax_4_1.grid('both')

  plt.tight_layout() 
  plt.close()

  ## SAMPLE FROM THE MODEL
  generated_data, fig_objects, fig_names = generate_data_and_plot(model, LATENT_DIM, fig_objects, fig_names, -1, 128, use_cuda)

  ## PCA ANALYSIS
  # Incremental PCA
  n_components = 2
  ipca = IncrementalPCA(n_components=n_components, batch_size=BATCH_SIZE)

  # Fit IPCA on training data
  for data in train_dl:
    data = data[:,:input_dim] 
    ipca.partial_fit(data.cpu().numpy())

  # Transform generated data
  generated_data_transformed = ipca.transform(generated_data)

  # Transform original data (in batches)
  def transform_in_batches(dataloader, pca_model):
    transformed_data_list = []
    for data in dataloader:
      data = data[:,:input_dim]
      transformed_batch = pca_model.transform(data.cpu().numpy())
      transformed_data_list.append(transformed_batch)
    return np.concatenate(transformed_data_list, axis=0)

  original_data_transformed = transform_in_batches(train_dl, ipca)

  fig = plt.figure(figsize=(5, 5))
  fig_objects.append(fig)
  fig_names.append("IPCA_orig_vs_generated_data")

  # Original data
  plt.scatter(original_data_transformed[:, 0], original_data_transformed[:, 1], alpha=0.7, label='Original Data')
  # Generated data
  plt.scatter(generated_data_transformed[:, 0], generated_data_transformed[:, 1], alpha=0.7, label='Generated Data')

  plt.title('PCA Proj. of Original and Generated GTEx-Gene Sequences')
  plt.xlabel('PC1')
  plt.ylabel('PC2')
  plt.grid('both')
  plt.legend()
  plt.close()
  
  ## OUTPUT DATA
  pca_output = PCA(n_components=n_components)
  pca_gen_data = pca_output.fit_transform(generated_data)

  fig = plt.figure(figsize=(5, 5))
  fig_objects.append(fig)
  fig_names.append("PCA_generated_data")

  plt.scatter(pca_gen_data[:, 0], pca_gen_data[:, 1], alpha=0.7, label='z')
  plt.title('PCA of generated data')
  plt.xlabel('PC1')
  plt.ylabel('PC2')
  plt.legend()
  plt.close()
  

  ## PCA ANALYSIS on latent representation

  n_components = 2

  pca = PCA(n_components=n_components)
  latent_pca = pca.fit_transform(latent_representations)

  fig = plt.figure(figsize=(5, 5))
  fig_objects.append(fig)
  fig_names.append("PCA_of_latent_representations")

  plt.scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.7)
  plt.title('PCA of latent representations')
  plt.xlabel('PC1')
  plt.ylabel('PC2')  
  plt.grid('both')
  plt.close()

  ## SAVE RESULTS
  save_results(f'{exp_file}{experiment_number}/parameters_{experiment_number}.txt', fig_objects, fig_names, variable_info, model)

######################################################################
# HELPER FUNCTIONS
######################################################################

######################################################################
def save_results(filename, fig_objects, fig_names, variable_info, model):

  # Extract directory from filename
  directory = os.path.dirname(filename)
  if not os.path.exists(directory):
    os.makedirs(directory)

  # Save variables
  with open(filename, 'w') as file:
    for var_name, var_value in variable_info:
      file.write(f"{var_name} = {var_value}\n")
    file.write("\n")

  # Save figures
  for fig, name in zip(fig_objects, fig_names):
    fig_path = os.path.join(directory, name + '.png')
    fig.savefig(fig_path)    

  # Save model  
  torch.save(model.state_dict(), f'{directory}/vae_model.pt')    
  print('Results saved.')

######################################################################
def generate_data_and_plot(model, LATENT_DIM, fig_objects, fig_names, epoch = -1, n_samples=128, use_cuda=False):
  
  model.eval()
  random_sequence = torch.randn(n_samples, LATENT_DIM)
  random_sequence = random_sequence.cuda() if use_cuda else random_sequence
  t_generated_data = model.sample(random_sequence)

  generated_data = t_generated_data.detach().cpu().numpy()

  sample_idx = 0

  fig, ax = plt.subplots(2, 1, figsize=(9,5))
  fig_objects.append(fig)
  fig_names.append(f'generated_data: epoch {epoch}')

  ax[0].plot(generated_data[sample_idx,:])
  ax[0].grid('both')
  ax[1].plot(generated_data[sample_idx  ,:], 'r')
  ax[1].plot(generated_data[sample_idx+1,:], 'b')
  ax[1].plot(generated_data[sample_idx+2,:], 'k')
  ax[1].plot(generated_data[sample_idx+3,:], 'g')
  ax[1].plot(generated_data[sample_idx+4,:], 'c')
  ax[1].set_xlim((0,100))
  ax[1].grid('both')
  plt.close()

  return generated_data, fig_objects, fig_names


