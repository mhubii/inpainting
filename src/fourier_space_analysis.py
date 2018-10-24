import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

# Perform fourier reconstruction with numpy.
rad = scipy.misc.imread('../autoencoded_inpainting/img/original.png')
inp = scipy.misc.imread('../autoencoded_inpainting/img/inpainted.png')

# Perform FFT.
ffts = [np.fft.fft2(rad), np.fft.fft2(inp)]

for idx, fft in enumerate(ffts):

    fshift = np.fft.fftshift(fft)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    ffts[idx] = magnitude_spectrum

    # Apply filter.
    center = [int(fft.shape[1] / 2), int(fft.shape[0] / 2)]

    y, x = np.ogrid[:fft.shape[0], :fft.shape[1]]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) * 10

    mask = dist_from_center >= min(fft.shape[0], fft.shape[1]) * 0.5


    rad = np.fft.ifftshift(fshift)
    rad = np.fft.ifft2(rad)
    rad = np.abs(rad)



    # Show results.
    plt.subplot(121),plt.imshow(rad, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

# Show difference image.
dif = ffts[0] - ffts[1]
plt.imshow(dif, cmap='gray')
plt.colorbar()
plt.show()

