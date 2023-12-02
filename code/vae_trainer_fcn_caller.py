import random
import math
import numpy as np
import sys
sys.path.append('../code')
from vae_trainer_fcn import vae_trainer_fcn
RANDOM_SEED=42
np.random.seed(RANDOM_SEED)
from datetime import datetime

#Hyperparameter options
# RANDOM SEARCH
if True:
  exp_file = 'results/exp_random_'
  exp_file = 'results/test_'
  batch_size_V            = [32]
  train_epoch_V           = [3, 5, 10]          # VARIABLE
  hidden_num_layers_V     = [1, 2, 3]           # VARIABLE
  latent_dim_V            = [4, 8, 16, 32, 64]  # VARIABLE
  LR_range                = [1e-4, 1e-3]        # VARIABLE
  LR_profile              = False
  beta_range              = [0.2, 10]           # VARIABLE
  max_latent_dim_exponent = 11  # 2^11 = 1024
  HIDDEN_LAYERS           = []                  # VARIABLE

# LR profiling, all other fixed
if False:
  exp_file = 'results/exp_lr_prof_'
  batch_size_V            = [32]
  train_epoch_V           = [10]
  hidden_num_layers_V     = [2]
  latent_dim_V            = [32]
  LR_range                = [1e-5, 1e-3] # 1e-3 z-space expansion
  LR_profile              = True
  beta_range              = [1, 1]
  HIDDEN_LAYERS           = [128, 1024] # FIXED

EXPERIMENTS = 200
LR_prof_V   = np.linspace(LR_range[0], LR_range[1], EXPERIMENTS)

for experiment_number in range(1, EXPERIMENTS+1):
  # Randomly select hyperparameters
  BATCH_SIZE        = random.choice(batch_size_V)
  TRAIN_EPOCHS      = random.choice(train_epoch_V) 
  BETA              = random.uniform(*beta_range)
  HIDDEN_NUM_LAYERS = random.choice(hidden_num_layers_V)
  LATENT_DIM        = random.choice(latent_dim_V)

  if (len(HIDDEN_LAYERS) == 0):
    latent_dim_exponent = math.floor(math.log(LATENT_DIM, 2))
    hidden_layers_V = (2**np.arange(latent_dim_exponent+1, max_latent_dim_exponent)).tolist()
    HIDDEN_LAYERS       = sorted(random.sample(hidden_layers_V, HIDDEN_NUM_LAYERS))
  else:
    # HIDDEN_LAYERS is fixed
    pass

  if LR_profile:
    LR          = LR_prof_V[experiment_number]  
    LR_status   = '(PROFILING)'
  else:
    LR          = random.uniform(*LR_range)
    LR_status   = ''

  # Call the training function
  current_time = datetime.now().strftime("%H:%M")
  print(f'\nEXPERIMENT: #{experiment_number}/{EXPERIMENTS} (Time: {current_time})')
  print('==============================')
  print(f'latent_dim      = {LATENT_DIM}')
  print(f'hidden layers   = {HIDDEN_LAYERS}')
  print(f'BATCH_SIZE      = {BATCH_SIZE}')
  print(f'TRAIN_EPOCHS    = {TRAIN_EPOCHS}')
  print(f'LEARNING RATE   = {LR:2.6f} {LR_status}')
  print(f'BETA            = {BETA:2.3f}')
  
  vae_trainer_fcn(BATCH_SIZE, TRAIN_EPOCHS, LR, BETA, HIDDEN_NUM_LAYERS, LATENT_DIM, HIDDEN_LAYERS, experiment_number, exp_file)
    
