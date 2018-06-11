import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.interpolate
import scipy.ndimage


# Load Radon Transform.
rad = scipy.misc.imread('../autoencoded_inpainting/img/original.png')


# Perform 1D Fast Fourier Transform.
fft = np.fft.fft(np.fft.ifftshift(rad, axes=0), axis=0)
fft_shift = np.fft.fftshift(fft, axes=0)


# Transform to 2D Fourier Space for the Slice Theorem.
radius = np.arange(rad.shape[0]) - 0.5*rad.shape[0]
alpha = np.arange(0, rad.shape[1], 1)*np.pi/180

radius, alpha = np.meshgrid(radius, alpha)


# Convert to Cartesian Coordinates.
radius = radius.flatten()
alpha = alpha.flatten()

src_x = 0.5*rad.shape[0] + radius*np.cos(alpha)
src_y = 0.5*rad.shape[0] + radius*np.sin(alpha)


# Interpolate to regular grid.
dst_x, dst_y = np.meshgrid(np.arange(rad.shape[0]), np.arange(rad.shape[0]))

dst_x = dst_x.flatten()
dst_y = dst_y.flatten()

fft2 = scipy.interpolate.griddata(
    (src_x, src_y),
    fft_shift.T.flatten(),
    (dst_x, dst_y),
    method='cubic',
    fill_value=0.0).reshape(rad.shape[0], rad.shape[0])


# Perform 2D inverse Fast Fourier Transform.
reco = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fft2)))


# Show results.
plt.subplot(221)
plt.imshow(rad, cmap='gray')
plt.title('Radon Transform')
plt.axis('off')

plt.subplot(222)
plt.imshow(np.log(np.abs(fft_shift)), cmap='gray')
plt.title('1D Fourier Transform')
plt.axis('off')

plt.subplot(223)
plt.imshow(np.log(np.abs(fft2)), cmap='gray')
plt.title('Mapping to 2D Fourier Space')
plt.axis('off')

plt.subplot(224)
plt.imshow(np.real(reco), cmap='gray')
plt.title('2D Inverse Fourier Transform')
plt.axis('off')

plt.savefig('../img/fourier_reconstruction.png')
