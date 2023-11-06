import torch
import os
import sys
from typing import List



def indexer(filename):
    # open file with byteread
    filesize = os.stat(filename).st_size
    try:
        infile = open(filename, "rb")
    except IOError as err:
        sys.exit("Cant open file:" + str(err))

    # initiate flags and position in file
    index_list = []
    pos = 0
    dp_start = 0
    first_flag = True
    chunk_size = 64**3
    while True:
        print(pos, filesize)
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

        if len(chunk) < chunk_size:
            break

        pos += chunk_size      # keep track of position in file
    infile.close()
    print(len(index_list))
    print(index_list)


class StreamDataLoader:
    def __init__(self, filename, batch_size, use_cuda=False):
        self.filename = filename
        self.batch_size = batch_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
    
    def __iter__(self):
        with open(self.filename, "r") as infile:
            header = infile.readline()
            batch = []
            for line in infile:
                
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
            
    
    def load_data_point(self, line: str) -> List:
        return [float(x) for x in line.strip().split("\t")[1:]]
    
    
    def process_batch(self, batch) -> torch.Tensor:
        return torch.tensor(batch, dtype=torch.float)


if __name__ == "__main__":
    filename = "data/archs_gene_very_small.tsv"
    
    dataloader = StreamDataLoader(filename, 10)
    
    for batch in dataloader:
        print(batch)
    