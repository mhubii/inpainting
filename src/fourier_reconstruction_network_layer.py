import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import torch
import torch.nn as nn
from torch.autograd import Variable

"""

Currently dependent on the implementations of the fftshift and
ifftshift functions in pytorch, which are not implemented yet.

"""


# Fourier reconstruction network.
"""
class FourierReconstruction(nn.Module):

    def __init__(self):

        super(FourierReconstruction, self).__init__()

    def forward(self, input):

        # Perform 1D Fast Fourier Transform.
        fft1 = torch.fft(input)
        
        # Perform coordinate transformation.
        
        # Interpolate to regular 2D Fourier space grid.
        
        # Perform 2D inverse Fast Fourier Transform.
"""


# Load radon transform and convert it.
rad = scipy.misc.imread('../img/rand_ell_rad.png')


# Perform reference fft.
fftnp = np.fft.fft(rad, axis=0)
plt.imshow(np.log(np.abs(fftnp)))
plt.show()


# Convert to torch tensor.
rad = rad.T
rad = torch.from_numpy(rad).float()
rad = rad.cuda()


# Pass radon transform through fourier reconstruction layer.
fft1 = torch.rfft(rad, 1, onesided=False)


# Convert resulting tensor to numpy array.
fft1 = fft1.cpu().numpy()
fft1 = np.transpose(fft1, (1, 0, 2))


# Show results.
#fft1 = np.transpose(fft1, (2, 0, 1))
print(fft1.shape)
#fft1 = np.abs(fft1)
plt.imshow(np.log(np.sqrt(np.square(fft1[:, :, 0]) + np.square(fft1[:, :, 1]))))
plt.show()


# There is currently no implementation of fftshift that support autograd.


# Transform from cylindrical to cartesian coordinates.



# Perform 2D inverse Fourier Transform.
