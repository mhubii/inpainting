import numpy as np
from skimage.transform import radon
import scipy.misc

# Set the resolution of the volume and the number of volumes.
nx = 64
ny = 64
nVol = 1000

# Load data.
data = np.load('Data/randVol1k.npy')
data.reshape([nVol, nx, ny])

# Perform the radon transformation.
a = 360
na = 64

angles = np.linspace(0., a, na, endpoint=False)

rad = []

for n in range(nVol):
    rad.append(radon(data[n], angles, circle=True))

data = np.asarray(rad).reshape([nVol, rad[0].shape[0], rad[0].shape[1]])

# Save data.
np.save('Data/radonRandVol1kGroundTruth', data)

scipy.misc.imsave('Images/randEllRad.png', data[0])
