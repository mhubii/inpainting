from torch.utils.data import Dataset
import scipy.misc
import csv
import torch.nn as nn

# Generally used parameters.
N_X = 1024
N_Y = 1024
N_VOL = 1

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

    def __init__(self, csv_loc):
        self.csv_loc = list()

        with open(csv_loc, 'r') as my_file:
            reader = csv.reader(my_file)
            self.csv_loc = list(reader)

    def __getitem__(self, index):
        loc = self.csv_loc[index][0]
        snippet = scipy.misc.imread(loc)

        return snippet

    def __len__(self):
        return len(self.csv_loc)


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0., 0.02)
