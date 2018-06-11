import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.interpolate
import scipy.ndimage
import math


# Load Radon Transform.
rad = scipy.misc.imread('/home/martin/Documents/inpainting/autoencoded_inpainting/img/original.png')

"""
# Create Radon Transform.
# Convenience function
def sqr(x): return x*x

S = 256

# Prepare a target image
x, y = np.meshgrid(np.arange(S)-S/2, np.arange(S)-S/2)
mask = (sqr(x)+sqr(y) <= sqr(S/2-10))
target = np.where(
    mask,
    scipy.misc.imresize(
        scipy.misc.face()[:, :, 0],
        (S, S),
        interp='cubic'
        ),
    np.zeros((S, S))
    )/255.0

plt.imshow(target)
plt.show()

N = 360


def angle(i): return (math.pi*i)/N


rad = np.array([
        np.sum(
            scipy.ndimage.interpolation.rotate(
                target,
                np.rad2deg(angle(i)), # NB rotate takes degrees argument
                order=3,
                reshape=False,
                mode='constant',
                cval=0.0
                )
            ,axis=0
            ) for i in range(N)
        ])

plt.imshow(rad)
plt.show()
"""

# Perform 1D Fast Fourier Transform.
fft = np.fft.fft(np.fft.ifftshift(rad, axes=0), axis=0)
fft_shift = np.fft.fftshift(fft, axes=0)
#fft_shift[int(fft_shift.shape[0]*0.5)-5:int(fft_shift.shape[0]*0.5)+5, :] = 1
magnitude = np.log(np.abs(fft_shift))

"""
# Coordinates of sinogram FFT-ed rows' samples in 2D FFT space
a = np.array([angle(i) for i in range(N)])
r = np.arange(S)-S/2
r, a = np.meshgrid(r, a)
r = r.flatten()
a = a.flatten()
src_x = (S/2)+r*np.cos(a)
src_y = (S/2)+r*np.sin(a)

# Coordinates of regular grid in 2D FFT space
dst_x, dst_y = np.meshgrid(np.arange(S), np.arange(S))
dst_x = dst_x.flatten()
dst_y = dst_y.flatten()

fft2 = scipy.interpolate.griddata(
    (src_y,src_x),
    fft_shift.flatten(),
    (dst_y,dst_x),
    method='cubic',
    fill_value=0.0
    ).reshape((S, S))


"""
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

plt.imshow(np.log(np.abs(fft2)))
plt.colorbar()
plt.show()

plt.imshow(np.real(reco))
plt.colorbar()
plt.show()

"""
# Show results.
plt.subplot(121), plt.imshow(rad)
plt.title('Radon Transform'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude)
plt.title('1D Fourier Transform'), plt.xticks([]), plt.yticks([])
plt.show()
"""

"""
# Create reference.
org_reco = scipy.misc.imread('/home/martin/Documents/inpainting/autoencoded_inpainting/img/original_reco.png')
org_fft2 = np.fft.fftshift(np.fft.fft2((org_reco)))
plt.imshow(np.log(np.abs(org_fft2)))
plt.colorbar()
plt.show()

org_reco = np.fft.ifft2(org_fft2)
plt.imshow(np.abs(org_reco))
plt.colorbar()
plt.show()
"""
