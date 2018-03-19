import numpy as np
import scipy.misc
import torch
from torch.autograd import Variable
from tqdm import tqdm
import model
import utils


# Load generator and discriminator.
gen = model.Generator(128).cuda()
gen.load_state_dict(torch.load('../state_dict/gen_epoch_20.pth'))

dis = model.Discriminator(128).cuda()
dis.load_state_dict(torch.load('../state_dict/dis_epoch_20.pth'))


# Load radon transform and normalize.
radon = scipy.misc.imread('../img/rand_ell_rad.png')
radon = radon/127.5 - 1.


# Apply mask.
mask = utils.create_mask(8)
masked_radon = np.multiply(radon, mask)


# Slice into sub-arrays.
stride = (48, 48)

radon_snippets, n_row_slice, n_col_slice, = utils.slice_array(radon, utils.IMAGE_HEIGHT, utils.IMAGE_WIDTH,
                                                              stride)

mask_snippets, _, _ = utils.slice_array(mask, utils.IMAGE_HEIGHT, utils.IMAGE_WIDTH,
                                        stride)

masked_radon_snippets, _, _ = utils.slice_array(masked_radon, utils.IMAGE_HEIGHT, utils.IMAGE_WIDTH,
                                                stride)


# Turn sub-arrays into a huge tensor.
n_snippets = len(radon_snippets)

for i in range(n_snippets):
    radon_snippets[i] = radon_snippets[i].reshape([1, 1, utils.IMAGE_HEIGHT, utils.IMAGE_WIDTH])
    mask_snippets[i] = mask_snippets[i].reshape([1, 1, utils.IMAGE_HEIGHT, utils.IMAGE_WIDTH])
    masked_radon_snippets[i] = masked_radon_snippets[i].reshape([1, 1, utils.IMAGE_HEIGHT, utils.IMAGE_WIDTH])

radon_snippets = np.concatenate(radon_snippets)
radon_snippets = torch.from_numpy(radon_snippets).float()
radon_snippets = Variable(radon_snippets).cuda()

mask_snippets = np.concatenate(mask_snippets)
mask_snippets = torch.from_numpy(mask_snippets).float()
mask_snippets = Variable(mask_snippets).cuda()

masked_radon_snippets = np.concatenate(masked_radon_snippets)
masked_radon_snippets = torch.from_numpy(masked_radon_snippets).float()
masked_radon_snippets = Variable(masked_radon_snippets).cuda()


# Perform the inpainting.
labels = torch.ones(n_snippets, 1, 1, 1)
labels = Variable(labels).cuda()


# Random input to optimize for.
rand_input = torch.randn(n_snippets, 100, 1, 1).cuda()
rand_input = Variable(rand_input, requires_grad=True)

optimizer = torch.optim.Adam([rand_input], lr=0.01)


# Factor to balance between prior and context loss.
lam = 0.003

for _ in tqdm(range(100)):
    optimizer.zero_grad()

    out_gen = gen(rand_input)
    context_loss = utils.context_loss(out_gen, radon_snippets, mask_snippets)

    out_dis = dis(out_gen)
    prior_loss = utils.prior_loss(out_dis, labels)

    loss = context_loss + lam*prior_loss

    loss.backward()
    optimizer.step()


# Stick snippets together.
opt = gen(rand_input)

opt = opt.cpu().data.numpy()
ori = radon_snippets.cpu().data.numpy()
mas = masked_radon_snippets.cpu().data.numpy()

opt = utils.stick_together(opt, n_row_slice, n_col_slice,
                           stride)

ori = utils.stick_together(ori, n_row_slice, n_col_slice,
                           stride)

mas = utils.stick_together(mas, n_row_slice, n_col_slice,
                           stride)


# Save results.
scipy.misc.imsave('../img/original.png', ori)

scipy.misc.imsave('../img/masked.png', mas)

scipy.misc.imsave('../img/optimal.png', opt)
