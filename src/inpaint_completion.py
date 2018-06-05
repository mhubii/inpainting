import numpy as np
import scipy.misc
import torch
from torch.autograd import Variable
from tqdm import tqdm
import model_completion
import utils


# Load completion network.
com = model_completion.Completion().cuda()
com.load_state_dict(torch.load('../state_dict/com_epoch_12.pth'))


# Load radon transform and normalize.
radon = scipy.misc.imread('../img/rand_ell_rad.png')
radon = radon/255


# Apply mask.
mask = utils.create_inv_mask(4)
masked_radon = np.multiply(radon, (1 - mask))


# Fill corrupted part with mean of uncorrupted part.
ind = np.nonzero(1 - mask)
mean = np.mean(masked_radon[ind])
masked_radon = np.where((1 - mask), masked_radon, mean)


# Expand dimensions.
masked_radon = np.expand_dims(masked_radon, 0)
masked_radon = np.expand_dims(masked_radon, 0)
mask = np.expand_dims(mask, 0)
mask = np.expand_dims(mask, 0)


# Perform the inpainting.
masked_radon = torch.from_numpy(masked_radon).float()
masked_radon = Variable(masked_radon).cuda()

mask = torch.from_numpy(mask).float()
mask = Variable(mask).cuda()

inp = com(masked_radon, mask)


# Convert to numpy.
masked_radon = masked_radon.cpu().data.numpy()
inp = inp.cpu().data.numpy()


# Squeeze dimensions.
masked_radon = np.squeeze(masked_radon, 0)
masked_radon = np.squeeze(masked_radon, 0)
inp = np.squeeze(inp, 0)
inp = np.squeeze(inp, 0)


# Save results.
scipy.misc.imsave('../img/completion_original.png', radon)
scipy.misc.imsave('../img/completion_masked.png', masked_radon)
scipy.misc.imsave('../img/completion_inpainted.png', inp)
