"""
    Doc 2017.09.29

    RadonTransform can be used to determine the radon transformation of given
    volumes.

"""

import numpy as np
from skimage.transform import radon

# set some initial parameters
nx = 128
ny = 128
nImg = 1000

# load data
data = np.load('randVol1k.npy')
data.reshape([nImg, nx, ny])

# perform the radon transformation
aTot = 360
na = 360

angles = np.linspace(0., aTot, na, endpoint=False)

rad = []

for n in range(nImg):
    rad.append(radon(data[n], angles, circle=True))

# save data
np.save('radonRandVol1k', rad)
