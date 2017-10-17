"""
    Doc 2017.09.29

    RadonTransform can be used to determine the radon transformation of given
    volumes.

"""

import numpy as np
from skimage.transform import radon

# set the resolution of the volume and the number of volumes.
nx = 256
ny = 512
nVol = 10

# load data
data = np.load('randVol1k.npy')
data.reshape([nVol, nx, ny])

# perform the radon transformation
a = 360
na = 360

angles = np.linspace(0., a, na, endpoint=False)

rad = []

for n in range(nVol):
    rad.append(radon(data[n], angles, circle=True))

data = np.asarray(rad).reshape([nVol, rad[0].shape[0], rad[0].shape[1]])

# mask data (reduce the scanned angle and the number of images)
aRed = 360
naRed = 90
daRed = int(aRed / naRed)

redData = np.zeros([nVol, rad[0].shape[0], rad[0].shape[1]])

for n in range(nVol):
    redData[n, :, 0:aRed:daRed] = data[n, :, 0:aRed:daRed]

# save data
np.save('radonRandVol1kGroundTruth', data)
np.save('radonRandVol1kReduced', redData)
