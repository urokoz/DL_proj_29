import torch
from typing import List


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
    dataloader = StreamDataLoader("archs_gene_very_small.tsv", 10)
    
    for batch in dataloader:
        print(batch)
    