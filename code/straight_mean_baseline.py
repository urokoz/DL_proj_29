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
from torch.utils.data import DataLoader


dat_dir = "data/hdf5"
archsDset = Archs4GeneExpressionDataset(data_dir = dat_dir, normalize=False, load_in_mem=False)
archs4_dataloader = DataLoader(archsDset, batch_size=100, num_workers=2, prefetch_factor=1)


criterion = torch.nn.MSELoss()
losses = []
for U in tqdm(archs4_dataloader):
    mean_BL = torch.tensor(np.array([archsDset.data_mean for _ in range(len(U))]))
    loss = criterion(U, mean_BL)
    losses.append(loss)
    
print("Straight mean loss:", np.mean(losses))

archsDset = Archs4GeneExpressionDataset(data_dir = dat_dir, normalize=True, load_in_mem=False)
archs4_dataloader = DataLoader(archsDset, batch_size=100, num_workers=2, prefetch_factor=1)

losses = []
for U in tqdm(archs4_dataloader):
    mean_BL = torch.tensor(np.array([archsDset.norm_mean for _ in range(len(U))]))
    loss = criterion(U, mean_BL)
    losses.append(loss)
    
print("Normalized mean loss:", np.mean(losses))

gtexDset = GtexDataset(data_dir=dat_dir, normalize=False, load_in_mem=True)
gtex_dataloader = DataLoader(gtexDset, batch_size=100, num_workers=2, prefetch_factor=1)  

losses = []
for L, I in tqdm(gtex_dataloader):
    mean_BL = torch.tensor(np.array([gtexDset.iso_mean for _ in range(len(I))]))
    print(I)
    print(mean_BL)
    loss = criterion(I, mean_BL)
    losses.append(loss)

print(losses)
print("Isoform mean loss:", np.mean(losses))