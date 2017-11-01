import numpy as np
import scipy.misc

sinogram = np.load('Data/genSin.npy')

sinogram = sinogram.reshape([100, 64, 64])

print(sinogram.shape)

scipy.misc.imsave('Data/genSin.png', sinogram[5])
