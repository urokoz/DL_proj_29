import random
import math
import numpy as np
import sys
from data_loader import Archs4GeneExpressionDataset
from torch.utils.data import DataLoader
sys.path.append('../code')
from vae_trainer_fcn import vae_trainer_fcn
RANDOM_SEED=42
np.random.seed(RANDOM_SEED)
from datetime import datetime

def generate_LR_and_beta(profile_LR, profile_beta, lr_values, beta_values, experiment_num):
    if profile_LR and not profile_beta:
        return lr_values[experiment_num], random.uniform(*beta_range)
    elif profile_beta and not profile_LR:
        return random.uniform(*LR_range), beta_values[experiment_num]
    elif profile_LR and profile_beta:
        lr_index = experiment_num // profile_entries
        beta_index = experiment_num % profile_entries
        return lr_values[lr_index], beta_values[beta_index]
    else:
        return random.uniform(*LR_range), random.uniform(*beta_range)

EXPERIMENTS     = 200 # number of experiments if no variable is profiled
profile_entries = 15  # number of experiments for each variable to be profiled
BATCH_SIZE      = 32 
NUM_WORKERS     = 2
PREFETCH_FACTOR = 1

if False:
  exp_file = 'results/exp_random_'    
  train_epoch_V           = [3, 5, 10]          # variable
  hidden_num_layers_V     = [1, 2, 3]           # variable
  latent_dim_V            = [4, 8, 16, 32, 64]  # variable
  LR_range                = [1e-4, 1e-3]        # variable
  LR_profile              = False               # Do not profile learning rate  
  beta_range              = [0.2, 10]           # variable  
  beta_profile            = False               # Do not profile beta
  HIDDEN_LAYERS           = []                  # variable
  max_latent_dim_exponent = 11  # 2^11 = 1024

# Beta profiling, all other fixed
elif True:
  exp_file = 'results/exp_beta_prof_'  
  train_epoch_V           = [10]                # fixed
  hidden_num_layers_V     = [2]                 # fixed
  latent_dim_V            = [32]                # fixed
  LR_range                = [0.5e-3, 0.5e-3]    # fixed
  LR_profile              = False               # Do not profile learning rate  
  beta_range              = [0.01, 2]           # fixed
  beta_profile            = True                # DO profile beta  
  HIDDEN_LAYERS           = [128, 1024]         # fixed
  max_latent_dim_exponent = 0                   # not used (HIDDEN_LAYERS is fixed)


# LR profiling, all other fixed
elif False:
  exp_file = 'results/exp_lr_prof_'  
  train_epoch_V           = [10]                # fixed
  hidden_num_layers_V     = [2]                 # fixed
  latent_dim_V            = [32]                # fixed
  LR_range                = [1e-5, 1e-3]        # VARIABLE 
  LR_profile              = True                # DO profile learning rate  
  beta_range              = [1, 1]              # fixed
  beta_profile            = False               # Do not profile beta  
  HIDDEN_LAYERS           = [128, 1024]         # fixed
  max_latent_dim_exponent = 0                   # not used (HIDDEN_LAYERS is fixed)


# {LR, beta} profiling, all other fixed
elif False:
  exp_file = 'results/lr_beta_prof_'
  train_epoch_V           = [10]                # fixed
  hidden_num_layers_V     = [2]                 # fixed
  latent_dim_V            = [32]                # fixed
  LR_range                = [1e-5, 1e-3]        # VARIABLE 
  LR_profile              = True                # DO profile learning rate
  beta_range              = [0.1, 10]           # VARIABLE
  beta_profile            = True                # DO profile beta  
  HIDDEN_LAYERS           = [128, 1024]         # fixed
  max_latent_dim_exponent = 0                   # not used (HIDDEN_LAYERS is fixed)


# Only relevant for variable profiling:
LR_prof_V   = np.linspace(np.log10(LR_range[0]), np.log10(LR_range[1]), profile_entries)
beta_prof_V = np.linspace(np.log10(beta_range[0]), np.log10(beta_range[1]), profile_entries)
# Convert back from log-space to working values
LR_prof_V   = 10 ** LR_prof_V
beta_prof_V = 10 ** beta_prof_V

#### Data loader setup ####  
dat_dir = "data/hdf5"
archsDset_train = Archs4GeneExpressionDataset(data_dir = dat_dir, split="train", load_in_mem=False)
train_dl = DataLoader(archsDset_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)
 
archsDset_val = Archs4GeneExpressionDataset(data_dir = dat_dir, split="val", load_in_mem=False)
val_dl = DataLoader(archsDset_val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)

# Number of experiments based on profiling
if LR_profile or beta_profile:
    EXPERIMENTS = profile_entries ** (int(LR_profile) + int(beta_profile))

for experiment_number in range(EXPERIMENTS):
  # Randomly select hyperparameters  
  TRAIN_EPOCHS      = random.choice(train_epoch_V)  
  HIDDEN_NUM_LAYERS = random.choice(hidden_num_layers_V)
  LATENT_DIM        = random.choice(latent_dim_V)

  if len(HIDDEN_LAYERS) == 0:
    latent_dim_exponent = math.floor(math.log(LATENT_DIM, 2))
    hidden_layers_V     = (2 ** np.arange(latent_dim_exponent + 1, max_latent_dim_exponent)).tolist()
    HIDDEN_LAYERS       = sorted(random.sample(hidden_layers_V, HIDDEN_NUM_LAYERS))
    
  # Generate LR and BETA based on the current setup
  LR, BETA = generate_LR_and_beta(LR_profile, beta_profile, LR_prof_V, beta_prof_V, experiment_number)  

  # Call the training function
  current_time = datetime.now().strftime("%H:%M")
  print(f'\nEXPERIMENT: #{experiment_number}/{EXPERIMENTS-1} (Time: {current_time})')
  print('==============================')
  print(f'latent_dim      = {LATENT_DIM}')
  print(f'hidden layers   = {HIDDEN_LAYERS}')  
  print(f'TRAIN_EPOCHS    = {TRAIN_EPOCHS}')
  print(f'LEARNING RATE   = {LR:2.6f}   profiling[{LR_profile}]')
  print(f'BETA            = {BETA:2.3f} profiling[{beta_profile}]')

  vae_trainer_fcn(BATCH_SIZE, NUM_WORKERS, PREFETCH_FACTOR, TRAIN_EPOCHS, LR, BETA, HIDDEN_NUM_LAYERS, LATENT_DIM, HIDDEN_LAYERS, experiment_number, exp_file, train_dl, val_dl)