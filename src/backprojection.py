import torch
import torch.nn as nn
import torch.nn.functional as f
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

rt = scipy.misc.imread('/home/martin/Documents/inpainting/autoencoded_inpainting/data/radon_transforms/radon_n_vol_0.jpg')

# filter

# backprojection
x = np.arange(rt.shape[0])
y = np.arange(rt.shape[0])
xg, yg = np.meshgrid(x, y)
theta = rt.shape[1]

rt = torch.from_numpy(rt)
bp = torch.zeros(rt.shape)

for t in range(theta):

    r = x*np.cos(t) + y*np.sin(t)

    # sample radon transform
    sa = torch.grid_sample(rt[[:,:,:,t]], r)

    # backproject 
    bp[xg, yg] += sa
