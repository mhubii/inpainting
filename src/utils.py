import numpy as np
import scipy.misc
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import skimage.transform


# Generally used parameters.
N_X = 256
N_Y = 256
N_VOL = 10000

A = 360
N_A = 360

IMAGE_HEIGHT = 63
IMAGE_WIDTH = 63


# Data set to get radon transform samples.
class RadonTransforms(Dataset):
    """
        Data set generator to obtain radon transformations.

    """

    def __init__(self, csv_loc, transform=None):
        self.csv_loc = list()
        self.transform = transform

        with open(csv_loc, 'r') as my_file:
            reader = csv.reader(my_file)
            self.csv_loc = list(reader)

    def __getitem__(self, index):
        loc = self.csv_loc[index][0]
        radon_transform = scipy.misc.imread(loc)
        radon_transform = np.expand_dims(radon_transform, 2)

        if self.transform is not None:
            radon_transform = self.transform(radon_transform)

        return radon_transform

    def __len__(self):
        return len(self.csv_loc)


# Data set to get radon snippet samples.
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


# Create a mask.
def create_mask(n):
    """
        Sets every every nth column of the mask to be
        one and return it.

    """
    mask = np.zeros([N_X, N_A])
    mask[:, ::n] = 1

    return mask


# Losses.
L1 = nn.L1Loss().cuda()
BCE = nn.BCELoss().cuda()


# Prior loss.
def prior_loss(inp, tar):
    loss = BCE(inp, tar)

    return loss


# Context loss.
def context_loss(inp, tar, mask):
    inp = torch.mul(inp, mask)
    loss = L1(inp, tar)

    return loss


# Slice array.
def slice_array(arr,
                n_rows, n_cols,
                stride):
    """
        Slices an array into multiple sub-arrays of size
        n_rows x n_cols and returns them as a list.

    """

    sliced_arr = list()

    # Determine how many sub-arrays can be obtained.
    n_row_slice = int(arr.shape[0]/stride[0])
    n_col_slice = int(arr.shape[1]/stride[1])

    for i in range(n_row_slice):
        for j in range(n_col_slice):
            if arr[i*stride[0]:i*stride[0] + n_rows,
                   j*stride[1]:j*stride[1] + n_cols].shape == (n_rows, n_cols):
                sliced_arr.append(arr[i*stride[0]:i*stride[0] + n_rows,
                                      j*stride[1]:j*stride[1] + n_cols])

    return sliced_arr, n_row_slice, n_col_slice


# Stick together.
def stick_together(sliced_arr, n_row_slice, n_col_slice,
                   stride):
    """
        Sticks sliced_arr together to return a full
        image.

    """

    n_rows, n_cols = sliced_arr[0, 0].shape

    arr = np.zeros([n_row_slice*stride[0] + n_rows - stride[0],
                    n_col_slice*stride[1] + n_cols - stride[1]])

    for i in range(n_row_slice):
        for j in range(n_col_slice):
            arr[i*stride[0]:i*stride[0] + n_rows,
                j*stride[1]:j*stride[1] + n_cols] = sliced_arr[i*n_col_slice + j]

    return arr


if __name__ == '__main__':
    mask = create_mask(2)
    scipy.misc.imsave('../data/mask.jpg', mask)
