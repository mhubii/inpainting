import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import model_completion
import utils


# Train policy.
def train(epoch, n_pre_train):
    for idx, data in enumerate(tqdm(data_loader)):

        data = Variable(data).cuda()

        # Randomly generate masks.
        mask = np.empty((args.batch_size, 1, data.shape[2], data.shape[3]))

        for i in range(args.batch_size):
            n = np.random.randint(2, 9)
            mask[i] = utils.create_inv_mask(n)

        mask = torch.from_numpy(mask).float()
        mask = Variable(mask).cuda()

        # Pre-train the completion network.
        if epoch < n_pre_train:

            # Optimize.
            opt_com.zero_grad()

            out_com = com(data, mask)

            loss_com = pre_loss(out_com, data)
            loss_com.backward()
            opt_com.step()

        # Perform adversarial training.
        else:

            # Train discriminator.
            opt_dis.zero_grad()

            real_label = Variable(torch.ones(args.batch_size)).cuda()
            fake_label = Variable(torch.zeros(args.batch_size)).cuda()

            out_com = com(data, mask)

            # Random local sample.
            n_x = np.random.randint(0, data.shape[2] - 128)
            n_y = np.random.randint(0, data.shape[3] - 128)

            in_local_real = data[:, :, n_x:n_x + 128, n_y:n_y + 128]
            in_local_fake = out_com[:, :, n_x:n_x + 128, n_y:n_y + 128]

            # Optimize.
            out_dis = dis(in_local_real, data)
            loss_dis = adv_loss(out_dis, real_label)

            out_dis = dis(in_local_fake, out_com)
            loss_dis += adv_loss(out_dis, fake_label)

            loss_dis.backward()
            opt_dis.step()

            # Train completion to fool discriminator.
            opt_com.zero_grad()

            loss_com = pre_loss(out_com, data) + adv_loss(out_dis, real_label)

            loss_com.backward()
            opt_com.step()

        if idx % 100 == 0:
            save_image(data.data, '../img/progress_completion/real_radon_transform_{}.png'.format(epoch + 1), nrow=3)
            save_image(torch.mul(data, 1 - mask).data, '../img/progress_completion/masked_radon_transform_{}.png'.format(epoch + 1), nrow=3)
            save_image(out_com.data, '../img/progress_completion/completed_radon_transform_{}.png'.format(epoch + 1), nrow=3)


if __name__ == '__main__':
    # Some initial settings
    parser = argparse.ArgumentParser(description='Training of completion network for inpainting.')
    parser.add_argument('-batch_size', type=int, default=6,
                        help='Batch size (default = 128).')
    parser.add_argument('-num_workers', type=int, default=1,
                        help='Number of workers to load data (default = 2).')
    parser.add_argument('-epochs', type=int, default=20,
                        help='Number of epochs (default = 10).')
    parser.add_argument('-csv_dir', type=str, default='../data/locations',
                        help='Location where the .csv file is stored that holds all file names (default = ../data/locations).')
    parser.add_argument('-mask_dir', type=str, default='../data/mask.png',
                        help='Location where the mask is stored (default = ../data/mask.png).')

    args = parser.parse_args()

    # Load completion network and discriminator.
    com = model_completion.Completion().cuda()
    dis = model_completion.Discriminator(args.batch_size).cuda()

    # Set up data loader.
    data_set = utils.RadonTransforms(args.csv_dir,
                                     transform=transforms.ToTensor())

    data_loader = torch.utils.data.DataLoader(data_set,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              drop_last=True)

    # Optimizer.
    opt_com = torch.optim.Adadelta(com.parameters())
    opt_dis = torch.optim.Adadelta(dis.parameters())

    # Loss for pre-training and adversarial loss.
    pre_loss = nn.MSELoss().cuda()
    adv_loss = nn.BCELoss().cuda()

    for epoch in range(args.epochs):
        train(epoch, 5)

        # Save results.
        torch.save(com.state_dict(), '../state_dict/com_epoch_{}.pth'.format(epoch + 1))
        torch.save(dis.state_dict(), '../state_dict/gl_dis.pth_epoch_{}.pth'.format(epoch + 1))
