import sys
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

gtexDset = GtexDataset(data_dir=dat_dir, normalize=False, load_in_mem=False)
gtex_dataloader = DataLoader(gtexDset, batch_size=100, num_workers=2, prefetch_factor=1)    

pca_model_path = f"archs_gtex_combined_pca.pkl"

if path.exists(pca_model_path):
    with open(pca_model_path, "rb") as f:
        ipca = pickle.load(f)
else:
    # create PCA incrementally
    ipca = IncrementalPCA(n_components=2)
    for sample in tqdm(archs4_dataloader, desc="Fitting PCA"):
        ipca.partial_fit(sample)
    for gtex_sample, _ in tqdm(gtex_dataloader, desc="Fitting PCA"):
        ipca.partial_fit(gtex_sample)
    # with open(pca_model_path, "wb") as f:
    #     pickle.dump(ipca, f)

# Transform original data (in batches)
def transform_in_batches(dataloader, pca_model):
    transformed_data_list = []
    for data in tqdm(dataloader, desc="Transforming data"):
        if len(data) == 2:
            data, _ = data
        transformed_batch = pca_model.transform(data.cpu().numpy())
        transformed_data_list.append(transformed_batch)
    return np.concatenate(transformed_data_list, axis=0)

ipca.transform(archsDset[:2])

archs4_data_transformed = transform_in_batches(archs4_dataloader, ipca)
gtex_data_transformed = transform_in_batches(gtex_dataloader, ipca)

fig = plt.figure(figsize=(5, 5))

plt.scatter(archs4_data_transformed[:, 0], archs4_data_transformed[:, 1], alpha=0.7, label='Archs4 Data')
plt.scatter(gtex_data_transformed[:, 0], gtex_data_transformed[:, 1], alpha=0.7, label='GTEx Data')

plt.title('PCA Projection of Archs4 and GTEx Gene Expressions')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
fig.savefig("IPCA_archs_vs_gtex_non_normalized.png")

