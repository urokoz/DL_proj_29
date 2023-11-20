import torch
import os
import sys
import random
import gzip
import numpy as np
import pickle
import json
from itertools import cycle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, BinaryIO


def openfile(filename, mode):
    """ Open gzip or normal files.
    """
    try:
        if filename.endswith('.gz'):
            fh = gzip.open(filename, mode=mode)
        else:
            fh = open(filename, mode)
    except:
        sys.exit("Can't open file:", filename)
    return fh


class StreamDataLoader(Dataset):
    def __init__(self, filename, batch_size=1, split=None, val_prop=0.2, target_file=None, use_cuda=False):
        self.filename = filename
        self.target_file = target_file
        self.batch_size = batch_size
        self.split = split
        self.val_prop = val_prop
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.positions = self.indexer(self.filename)
        
        val_split_index = int(np.floor(self.val_prop * len(self.positions)))
        if self.split == "train":
            self.positions = self.positions[val_split_index:]
        elif self.split == "val":
            self.positions = self.positions[:val_split_index]
        
        if self.target_file:
            self.target_positions = self.indexer(self.target_file)
            if self.split == "train":
                self.target_positions = self.target_positions[val_split_index:]
            elif self.split == "val":
                self.target_positions = self.target_positions[:val_split_index]
        else:
            self.target_positions = None 
        
        print("Dataset loaded")

    
    def __len__(self):
        return len(self.positions)
    
    
    def __getitem__(self, idx):
        (start, stop) = self.positions[idx]
        
        with openfile(self.filename, "rb") as infile:
            sample = np.loadtxt([self.extract(infile, start, stop)])
        sample = torch.tensor(sample, dtype=torch.float)
        
        if self.target_positions:
            (start, stop) = self.target_positions[idx]
            with open(self.target_file, "rb") as infile:
                targets = np.loadtxt([self.extract(infile, start, stop)])
            targets = torch.tensor(targets, dtype=torch.float)
        else:
            targets = torch.empty((0, 1))
        
        return (sample, targets)
        
    
    def get_uncompressed_size(self, file):
        pipe_in = os.popen('gzip -l %s' % file)
        list_1 = pipe_in.readlines()
        list_2 = list_1[1].split()
        c , u , r , n = list_2
        return int(u)
    
    
    def __iter__(self) -> torch.Tensor:
        """ Generator for getting batches out from the datasets.

        Yields:
            torch.Tensor: self.batch_size x n_featrues 
        """
        with openfile(self.filename, "rb") as infile:
            batch = []
            t = tqdm(total=len(self.positions))
            for i in range(0, len(self.positions), self.batch_size): 
                
                batch = self.load_batch(infile, i, self.batch_size)
                
                t.update(len(batch))
                batch = self.process_batch(batch)
                
                if self.use_cuda:
                    batch = batch.to(self.device)

                yield batch
        t.close()
    
    
    def load_batch(self, infile: BinaryIO, idx: int, step: int) -> np.ndarray:
        """Loads a batch of datapoints from given positions 
        in a file based on the index (idx) in the positions list.

        Args:
            infile (BinaryIO): datafile read in byteread mode
            idx (int): index for first data point position in the batch
            step (int): number of data points to include in the batch

        Returns:
            np.ndarray: batch of datapoints
        """
        batch = []
        for [start, stop] in self.positions[idx:idx+step]:
            dp = self.extract(infile, start, stop)
            batch.append(dp)
        return np.loadtxt(batch)
    
    
    def process_batch(self, batch: np.ndarray) -> torch.Tensor:
        """ Method to do preprocessing on the batch.
        For now it is just converting to a tensor.

        Args:
            batch (np.ndarray): The batch loaded in as a np.ndarray

        Returns:
            torch.Tensor: Preprocessed batch as a tensor ready to use
        """
        # place where we can normalize the batch? 
        # Or should the batches be normalized in relation to the entire dataset?
        return torch.tensor(batch, dtype=torch.float)


    def extract(self, buff: BinaryIO, start: int, stop:int) -> bytes:
        """ Extracts a section of a file based on start and stop indexes.

        Args:
            buff (BinaryIO): Filehandle reading in bytes modes  
            start (int): index of the start of the file section
            stop (int): index of the end of the file section

        Returns:
            bytes: bytes representation of the file section
        """
        buff.seek(start)
        return buff.read(stop-start)
    
    
    def indexer(self, filename: str) -> List:
        """ Indexes the data portions of the data file, 
        by finding the start and end of gene expression columns.
        It excludes the first column with the IDs.
        The indexing is specific to this file format.       

        Args:
            filename (str): Name of the tsv file containing the data.

        Returns:
            List: List of lists containing the indexes
        """
        # open file with byteread
        cache_file = filename+".index.pkl"
        
        if os.path.isfile(cache_file):
            with open(cache_file, "rb") as f:
                index_list = pickle.load(f)
            return index_list
        
        print(f"Indexing {filename}")
        filesize = os.stat(filename).st_size
        if filename.endswith(".gz") and filesize > 2**32:
            filesize = self.get_uncompressed_size(filename)
        elif filename == "data/archs4_gene_expression_norm_transposed.tsv.gz":
            filesize = 51732577723

        try:
            infile = openfile(filename, "rb")
            header = infile.readline()
        except IOError as err:
            sys.exit("Cant open file:" + str(err))

        t = tqdm(total=filesize-len(header))
        # initiate flags and position in file
        start_nums = True 
        end_nums = False
        index_list = []
        pos = len(header)
        chunk_size = 64**3
        while True:
            chunk = infile.read(chunk_size) # read chunk
            index = 0

            while True:     # seek through end of chunk
                if start_nums:
                    index = chunk.find(b"\t", index)
                    
                    if index == -1:
                        break
                    else:
                        dp_start = pos + index +1
                        start_nums = False 
                        end_nums = True

                if end_nums:
                    index = chunk.find(b"\n", index)
                    if index == -1:     # header not found
                        break
                    else:       # datapoint end found
                        dp_end = pos + index
                        index_list.append([dp_start, dp_end])
                        start_nums = True 
                        end_nums = False

            t.update(len(chunk))
            if len(chunk) < chunk_size:
                t.close()
                break

            pos += chunk_size      # keep track of position in file
        infile.close()
        
        with open(cache_file, "wb") as f:
            pickle.dump(index_list, f)
            
        return index_list


def get_dataset(data_dict: dict, batch_size: int, n_targets=None):
    """Takes a location for the datafiles and returns 3 dataloaders.
    The dataloaders return random indicies which means that it doesn't
    work well with gzipped files. 

    Args:
        data_dict (dict): Dict with paths to the datafiles.
        batch_size (int): number of samples per batch
        n_targets (int): Not implemented yet
    """
    # TODO: implement n_targets to be able to predict a subset of the targets
    unlabeled_data = StreamDataLoader(filename=data_dict["unlabeled"], use_cuda=True)
    training_data = StreamDataLoader(filename=data_dict["labeled_samples"], split="train", 
                                target_file=data_dict["labeled_targets"], use_cuda=True)
    validation_data = StreamDataLoader(filename=data_dict["labeled_samples"], split="val", 
                                  target_file=data_dict["labeled_targets"], use_cuda=True)
    
    unlabeled = DataLoader(unlabeled_data, batch_size=1000, num_workers=4, prefetch_factor=3)
    training = DataLoader(training_data, batch_size=batch_size, num_workers=4, prefetch_factor=3)
    validation = DataLoader(validation_data, batch_size=batch_size, num_workers=4, prefetch_factor=3)

    return unlabeled, training, validation

if __name__ == "__main__":
    with open("data/data.json", "r") as f:
        data_dict = json.load(f)
    
    unlabeled_dataloader, training_dataloader, validation_dataloader = get_dataset(data_dict=data_dict, batch_size=128) 
    
    testfile = open("test.txt", "w")
    t = tqdm(total=len(unlabeled_dataloader))
    for (x, y), (u, _) in zip(cycle(training_dataloader), unlabeled_dataloader):
        t.update()
        print(x.size(), y.size(), u.size(), file=testfile)
        continue
    t.close()
    
    # filename = "data/archs4_gene_small.tsv"
    # dataloader = StreamDataLoader(filename, split = "train", batch_size=64, use_cuda=True)    
    
    # dataloader_torch = DataLoader(dataloader, batch_size=64, num_workers=4, prefetch_factor=3)
    # # TODO: Understand the sampler and how it chooses data.
    
    # t = tqdm(total=len(dataloader_torch))
    # for batch in dataloader_torch:
    #     t.update()
    #     continue
    # t.close()
    
    # labeled_samples = "data/gtex_gene_expr.tsv"
    # labeled_targets = "data/gtex_iso_expr.tsv"
    # labeled_dataloader = StreamDataLoader(labeled_samples, train=True, target_file=labeled_targets, batch_size=64, use_cuda=True)
    
    # torch_dataloader = DataLoader(labeled_dataloader, batch_size=64, num_workers=4, prefetch_factor=3)
    
    # t = tqdm(total=len(torch_dataloader))
    # for batch in torch_dataloader:
    #     t.update()
    # t.close()

    # print(batch[0].size())
    # print(batch[1].size())