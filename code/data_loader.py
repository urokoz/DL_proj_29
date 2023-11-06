import torch
import os
import sys
import random
import numpy as np
from tqdm import tqdm
from typing import List


class StreamDataLoader:
    def __init__(self, filename, batch_size, use_cuda=False, shuffle_seed=None):
        self.filename = filename
        self.batch_size = batch_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.indexes = self.indexer(self.filename)
        if shuffle_seed:
            random.seed(shuffle_seed)
            random.shuffle(self.indexes)
        
    
    def __iter__(self):
        with open(self.filename, "rb") as infile:
            batch = []
            t = tqdm(total=len(self.indexes))
            for i in range(0, len(self.indexes), self.batch_size): 
                
                batch = self.load_batch(infile, i, self.batch_size)
                
                t.update(len(batch))
                batch = self.process_batch(batch)
                
                if self.use_cuda:
                    batch = batch.to(self.device)

                yield batch
        t.close()
    
    
    def load_batch(self, infile, idx, step):
        batch = []
        for [start, stop] in self.indexes[idx:idx+step]:
            batch.append(self.extract(infile, start, stop))
        return(np.loadtxt(batch))
    
    
    def indexer(self, filename):
        # open file with byteread
        filesize = os.stat(filename).st_size
        try:
            infile = open(filename, "rb")
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
        return index_list
    
    
    def process_batch(self, batch) -> torch.Tensor:
        return torch.tensor(np.array(batch), dtype=torch.float)


    def extract(self, buff, start, stop):
        buff.seek(start)
        return buff.read(stop-start)


if __name__ == "__main__":
    filename = "data/archs4_gene_small.tsv"
    
    dataloader = StreamDataLoader(filename, 64, shuffle_seed=42)    
    
    for batch in dataloader:
        continue
    