import numpy as np
from skimage.transform import iradon
import scipy.misc
import utils


# Load image.
# original = scipy.misc.imread('../img/original_rad.png')
# masked = scipy.misc.imread('../img/masked_rad.png')
# optimal = scipy.misc.imread('../img/optimal.png')

original = scipy.misc.imread('../img/completion_original.png')
masked = scipy.misc.imread('../img/masked_rad.png')
optimal = scipy.misc.imread('../img/completion_inpainted.png')


# Perform filtered back projection.
angles_original = np.linspace(0., original.shape[1], original.shape[1], endpoint=False)
angles_masked = np.linspace(0., utils.A, masked.shape[1], endpoint=False)
angles_optimal = np.linspace(0., optimal.shape[1], optimal.shape[1], endpoint=False)

original = iradon(original, angles_original, circle=True)
masked = iradon(masked, angles_masked, circle=True)
optimal = iradon(optimal, angles_optimal, circle=True)


# Save results.
# scipy.misc.imsave('../img/reco_original.png', original)
# scipy.misc.imsave('../img/reco_masked.png', masked)
# scipy.misc.imsave('../img/reco_optimal.png', optimal)

scipy.misc.imsave('../img/completion_reco_original.png', original)
scipy.misc.imsave('../img/completion_reco_masked.png', masked)
scipy.misc.imsave('../img/completion_reco_inpainted.png', optimal)
