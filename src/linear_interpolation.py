import numpy as np
import scipy.misc
from scipy import interpolate
import matplotlib.pyplot as plt

# Load radon transform.
rt = scipy.misc.imread('/home/martin/Downloads/inpainting/autoencoded_inpainting/img/masked.png')

# Split cause of memory issues.
upper_rt = rt[:128, :]
lower_rt = rt[128:, :]

# Create grids.
p, r = np.mgrid[0:upper_rt.shape[0]:1, 0:upper_rt.shape[1]:4]
ip_p, ip_r = np.mgrid[0:upper_rt.shape[0]:1, 0:upper_rt.shape[1]:1]

# Remove unkown values.
upper_rt = upper_rt[:, ::4]
lower_rt = lower_rt[:, ::4]

# Radial basis function interpolation.
f = interpolate.Rbf(p, r, upper_rt, function='linear')
ip_upper_rt = f(ip_p, ip_r)

f = interpolate.Rbf(p, r, lower_rt, function='linear')
ip_lower_rt = f(ip_p, ip_r)

ip_rt = np.concatenate((ip_upper_rt, ip_lower_rt), axis=0)

scipy.misc.imsave('../img/linear_interpolation.png', ip_rt)

