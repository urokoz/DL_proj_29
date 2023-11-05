# %% [markdown]
# #### Import libraries

# %%
import pandas as pd
import random
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import set_config
import matplotlib.pyplot as plt

# %% [markdown]
# #### Ensure reproducibility
# - To be expanded

# %%
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# %%
# Load data (remove header and first column / sample ids)
gene_data = pd.read_csv('archs_gene_very_small.tsv', sep='\t', header=None, skiprows=1)
gene_data = gene_data.iloc[:, 1:]
gene_data.head()

# %% [markdown]
# #### PCA + t-SNE

# %%
# PCA
# number of PCA components  must be less than the number of samples
n_PCA_components = min(20, min(gene_data.shape) - 1)
pca_obj = PCA(n_components=n_PCA_components)
pca_result = pca_obj.fit_transform(gene_data)

n_TSNE_components = 2 
n_TSNE_components = 3 

# perplexity must be less than the number of samples
perplexity_value = min(20, gene_data.shape[0] - 1)  
tsne_obj = TSNE(n_components=n_TSNE_components, random_state=SEED, perplexity=perplexity_value)
tsne_result = tsne_obj.fit_transform(pca_result)

# %% [markdown]
# #### Plots

# %%
if n_TSNE_components == 2:
    # 2D
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
else: 
    # 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_result[:, 0], tsne_result[:, 1], tsne_result[:, 2])
plt.show()


