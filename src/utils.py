from torch.utils.data import Dataset
import numpy as np
import scipy.misc
import csv
import torch.nn as nn

# Generally used parameters.
N_X = 1024
N_Y = 1024
N_VOL = 10

A = 360
N_A = 1024

IMAGE_HEIGHT = 63
IMAGE_WIDTH = 63


# Data set to get samples.
class RadonSnippets(Dataset):
    """
        Data set generator to obtain snippets of
        a radon transformation.

    """

    def __init__(self, csv_loc, transform=None):
        self.csv_loc = list()
        self.transform = transform

        with open(csv_loc, 'r') as my_file:
            reader = csv.reader(my_file)
            self.csv_loc = list(reader)

    def __getitem__(self, index):
        loc = self.csv_loc[index][0]
        snippet = scipy.misc.imread(loc)
        snippet = np.expand_dims(snippet, 2)

        if self.transform is not None:
            snippet = self.transform(snippet)

        return snippet

    def __len__(self):
        return len(self.csv_loc)
