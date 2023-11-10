import torch
import os
import sys
import numpy as np
import pickle
from tqdm import tqdm
from typing import List


class StreamDataLoader:
    def __init__(self, filename, batch_size, train=True, labeled=False, use_cuda=False):
        self.filename = filename
        self.batch_size = batch_size
        self.train = train
        self.labeled = labeled
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.indexes = self.indexer(self.filename)
        print("Dataset loaded")

    
    def __len__(self):
        return len(self.indexes)
    
    
    def __getitem__(self, idx):
        if self.labeled:
            pass
            # TODO: Implement the labeled logic
        else:
            pass
    
    
    def __iter__(self):
        with open(self.filename, "r") as infile:
            header = infile.readline()
            batch = []
            t = tqdm(total=len(self.indexes))
            for i, line in enumerate(infile):
                t.update()
                data_point = self.load_data_point(line)
                
                batch.append(data_point)
                
                if len(batch) == self.batch_size:
                    batch_tensor = self.process_batch(batch)
                    
                    if self.use_cuda:
                        batch_tensor = batch_tensor.to(self.device)
                
                    yield batch_tensor
                
                    batch = []
            
            if batch:
                batch_tensor = self.process_batch(batch)
                    
                if self.use_cuda:
                    batch_tensor = batch_tensor.to(self.device)
                
                yield batch_tensor  
        t.close()
                
    
    def indexer(self, filename):
        # open file with byteread
        cache_file = filename+".index.pkl"
        
        if os.path.isfile(cache_file):
            with open(cache_file, "rb") as f:
                index_list = pickle.load(f)
            return index_list
        
        print("Indexing file")
        filesize = os.stat(filename).st_size
        t = tqdm(total=filesize)
        try:
            infile = open(filename, "rb")
        except IOError as err:
            sys.exit("Cant open file:" + str(err))

        # initiate flags and position in file
        index_list = []
        pos = 0
        dp_start = 0
        chunk_size = 64**3
        while True:
            chunk = infile.read(chunk_size) # read chunk
            index = 0

            while True:     # seek through end of chunk
                index = chunk.find(b"\n", index+1)
                if index == -1:     # header not found
                    break
                else:       # datapoint end found
                    dp_end = pos + index - 1
                    index_list.append([dp_start, dp_end])

                    dp_start = pos + index
            t.update(len(chunk))
            if len(chunk) < chunk_size:
                break

            pos += chunk_size      # keep track of position in file
        infile.close()
        t.close()
        with open(cache_file, "wb") as f:
            pickle.dump(index_list, f)
        
        return index_list
            
    
    def load_data_point(self, line: str):
        # Convert the line to a NumPy array, skipping the first entry
        return np.fromstring(line.strip().split("\t", 1)[1], dtype=float, sep='\t')
    
    
    def process_batch(self, batch) -> torch.Tensor:
        return torch.tensor(np.array(batch), dtype=torch.float)


def get_dataset(location, batch_size, n_targets):
    """Takes a location for the datafiles and returns 3 dataloaders

    Args:
        location (_type_): _description_
        batch_size (_type_): _description_
        n_targets (_type_): _description_
    """
    



if __name__ == "__main__":
    filename = "data/archs4_gene_small.tsv"
    
    dataloader = StreamDataLoader(filename, 10)
    
    for batch in dataloader:
        continue
    