import torch
import os
import sys
import random
import gzip
import numpy as np
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


class StreamDataLoader:
    def __init__(self, filename, batch_size, use_cuda=False, shuffle_seed=None):
        self.filename = filename
        self.batch_size = batch_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.positions = self.indexer(self.filename)
        if shuffle_seed:    # insanely slow on gzipped files
            random.seed(shuffle_seed)
            random.shuffle(self.positions)
    
    
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
        return index_list


if __name__ == "__main__":
    filename = "data/archs4_gene_expression_norm_transposed.tsv.gz"
    
    dataloader = StreamDataLoader(filename, batch_size=10, use_cuda=True)    
    
    for batch in dataloader:
        continue
    print(batch)
