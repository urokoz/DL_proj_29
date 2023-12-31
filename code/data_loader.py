import h5py
import re
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import WeightedRandomSampler, DataLoader
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class Archs4GeneExpressionDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, split: str="", val_prop: float=0.1, normalize=False, load_in_mem: bool=False):
        print("# Loading ARCHS4 data...")
        f_archs4 = h5py.File(data_dir + '/archs4_gene_expression_norm_transposed.hdf5', mode='r')
        self.dset = f_archs4['expressions']
        self.normalize = normalize

        if load_in_mem:
            self.dset = np.array(self.dset)

        self.idxs = None
        
        if split == "train":
            if not self.idxs:
                self.idxs = np.arange(len(self.dset))
            
            self.idxs, _ = train_test_split(self.idxs, test_size=val_prop, random_state=42)
            self.idxs = np.sort(self.idxs)
            
        elif split == "val":
            if not self.idxs:
                self.idxs = np.arange(len(self.dset))
            
            _, self.idxs = train_test_split(self.idxs, test_size=val_prop, random_state=42)
            self.idxs = np.sort(self.idxs)
        
        self.data_mean = self.mean()
        self.data_std = self.std()
        self.norm_mean = self.normalized_mean()

    def __len__(self):
        if self.idxs is None:
            return self.dset.shape[0]
        else:
            return self.idxs.shape[0]

    def __getitem__(self, idx):
        if self.idxs is None:
            data = self.dset[idx]
        else:
            data = self.dset[self.idxs[idx]]
        if self.normalize:
            data = (data - self.data_mean) / self.data_std
        return data
        
    def mean(self):
        return np.mean(self.dset, axis=0)
    
    def std(self):
        return np.std(self.dset, axis=0)
    
    def normalized_mean(self):
        return np.mean((self.dset - self.data_mean) / self.data_std, axis=0)


class GtexDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, split: str="", val_prop: float=0.1, include: str="", exclude: str="", normalize=False, load_in_mem: bool=False):
        print("# Loading GTEx data...")
        f_gtex_gene = h5py.File(data_dir + '/gtex_gene_expression_norm_transposed.hdf5', mode='r')
        f_gtex_isoform = h5py.File(data_dir + '/gtex_isoform_expression_norm_transposed.hdf5', mode='r')

        self.dset_gene = f_gtex_gene['expressions']
        self.dset_isoform = f_gtex_isoform['expressions']
        
        self.normalize = normalize

        assert(self.dset_gene.shape[0] == self.dset_isoform.shape[0])

        if load_in_mem:
            self.dset_gene = np.array(self.dset_gene)
            self.dset_isoform = np.array(self.dset_isoform)

        self.idxs = None

        if include and exclude:
            raise ValueError("You can only give either the 'include_only' or the 'exclude_only' argument.")

        if include:
            matches = [bool(re.search(include, s.decode(), re.IGNORECASE)) for s in f_gtex_gene['tissue']]
            self.idxs = np.where(matches)[0]

        elif exclude:
            matches = [not(bool(re.search(exclude, s.decode(), re.IGNORECASE))) for s in f_gtex_gene['tissue']]
            self.idxs = np.where(matches)[0]
            
        if split == "train":
            if not self.idxs:
                self.idxs = np.arange(len(self.dset_gene))
            
            self.idxs, _ = train_test_split(self.idxs, test_size=val_prop, random_state=42, stratify=f_gtex_gene['tissue'][self.idxs])
            self.idxs = np.sort(self.idxs)
            
        elif split == "val":
            if not self.idxs:
                self.idxs = np.arange(len(self.dset_gene))
            
            _, self.idxs = train_test_split(self.idxs, test_size=val_prop, random_state=42, stratify=f_gtex_gene['tissue'][self.idxs])
            self.idxs = np.sort(self.idxs)
        
        self.sampleweights(f_gtex_gene['tissue'])
        self.gene_mean = self.gene_mean()
        self.gene_std = self.gene_std()
        self.iso_mean = self.isoform_mean()
        

    def sampleweights(self, labels_gene):
        #adding stuff here 
        #making up for imbalance in dataset by adding sampling weights. 
        if self.idxs is None:
            tissue_counts = Counter(labels_gene)
            self.sample_weights = [1/tissue_counts[i] for i in labels_gene]
        else:
            tissue_counts = Counter(labels_gene[self.idxs])
            self.sample_weights = [1/tissue_counts[i] for i in labels_gene[self.idxs]]
        
    def gene_mean(self):
        return np.mean(self.dset_gene, axis=0)
    
    def gene_std(self):
        return np.std(self.dset_gene, axis=0)
    
    def isoform_mean(self):
        return np.mean(self.dset_isoform, axis=0)

    def __len__(self):
        if self.idxs is None:
            return self.dset_gene.shape[0]
        else:
            return self.idxs.shape[0]

    def __getitem__(self, idx):
        if self.idxs is None:
            gene_data, isoform_data = self.dset_gene[idx], self.dset_isoform[idx]
        else:
            gene_data, isoform_data = self.dset_gene[self.idxs[idx]], self.dset_isoform[self.idxs[idx]]
        if self.normalize:
            gene_data = (gene_data - self.gene_mean) / self.gene_std
        return gene_data, isoform_data


if __name__ == '__main__':
    dat_dir = "data/hdf5"
    archsDset = Archs4GeneExpressionDataset(data_dir = dat_dir, load_in_mem=False)
    archsDloader = DataLoader(archsDset, batch_size=64, num_workers=2, prefetch_factor=1) 
    
    archsDset_train = Archs4GeneExpressionDataset(data_dir = dat_dir, split="train", load_in_mem=False)
    archsDloader_train = DataLoader(archsDset_train, batch_size=64, num_workers=2, prefetch_factor=1) 
    
    archsDset_val = Archs4GeneExpressionDataset(data_dir = dat_dir, split="val", load_in_mem=False)
    archsDloader_val = DataLoader(archsDset_val, batch_size=64, num_workers=2, prefetch_factor=1) 

    gtexDset = GtexDataset(data_dir = dat_dir, load_in_mem=True)
    sampler = WeightedRandomSampler(weights=gtexDset.sample_weights, num_samples=len(gtexDset), replacement=True)
    GtexDloader = DataLoader(gtexDset, sampler=sampler, batch_size=64, num_workers=2, prefetch_factor=1)
    
    gtexDset_train = GtexDataset(data_dir=dat_dir, split="train", load_in_mem=False)
    sampler_train = WeightedRandomSampler(weights=gtexDset_train.sample_weights, num_samples=len(gtexDset_train), replacement=True)
    GtexDloader_train = DataLoader(gtexDset_train, sampler=sampler_train, batch_size=64, num_workers=2, prefetch_factor=1)
    
    gtexDset_val = GtexDataset(data_dir=dat_dir, split="val", load_in_mem=False)
    sampler_val = WeightedRandomSampler(weights=gtexDset_val.sample_weights, num_samples=len(gtexDset_val), replacement=True)
    GtexDloader_val = DataLoader(gtexDset_val, batch_size=64, num_workers=2, prefetch_factor=1)
    
    # check that the train and validation sets are disjoint
    intersection = set(archsDset_train.idxs).intersection(set(archsDset_val.idxs))
    assert(len(intersection) == 0)
    intersection = set(gtexDset_train.idxs).intersection(set(gtexDset_val.idxs))
    assert(len(intersection) == 0)
    assert(len(gtexDset_train.idxs) + len(gtexDset_val.idxs) == len(gtexDset))

    for _ in tqdm(archsDloader):
        pass

    for _ in tqdm(archsDloader_train):
        pass

    for _ in tqdm(archsDloader_val):
        pass
    
    for _ in tqdm(GtexDloader_train):
        pass

    for _ in tqdm(GtexDloader_val):
        pass
