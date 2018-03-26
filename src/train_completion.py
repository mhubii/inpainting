import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import model_completion
import utils


# Train policy.
def train(epoch, n_pre_train):
    for idx, data in enumerate(tqdm(data_loader)):

        data = Variable(data).cuda()

        # Pre-train the completion network.
        if epoch < n_pre_train:
            # Randomly generate a mask.
            n = np.random.randint(2, 11)

            mask = utils.create_mask(n)
            mask = torch.from_numpy(mask).float()
            mask = Variable(mask).cuda()

            # Optimize.
            opt_com.zero_grad()
            # TODO take masked data as input and initialize masked regions with mean value.
            out = com(data)

            loss_com = pre_loss(torch.mul(out, mask), torch.mul(data, mask))
            loss_com.backward()
            opt_com.step()

        # Perform adversarial training.
        else:

            # Train discriminator.
            opt_dis.zero_grad()

            real_label = Variable(torch.ones(args.batch_size)).cuda()
            fake_label = Variable(torch.zeros(args.batch_size)).cuda()

            out = com(data)

            # Random local sample.
            n_x = np.random.randint(0, data.shape[2] - 128)
            n_y = np.random.randint(0, data.shape[3] - 128)

            in_local_real = data[:, :, n_x:n_x + 128, n_y:n_y + 128]
            in_local_fake = out[:, :, n_x:n_x + 128, n_y:n_y + 128]

            # Global sample.
            in_global_real = F.grid_sample(data, grid)
            in_global_fake = F.grid_sample(out, grid)

            # Optimize.
            out = dis(in_local_real, in_global_real)
            loss_dis = adv_loss(out, real_label)

            out = dis(in_local_fake, in_global_fake)
            loss_dis += adv_loss(out, fake_label)

            loss_dis.backward(retain_graph=True)
            opt_dis.step()

            # Randomly generate a mask.
            n = np.random.randint(2, 11)

            mask = utils.create_mask(n)
            mask = torch.from_numpy(mask).float()
            mask = Variable(mask).cuda()

            # Train completion to fool discriminator.
            opt_com.zero_grad()

            out_com = com(data)
            in_global_fake = F.grid_sample(out_com, grid)

            out = dis(in_local_fake, in_global_fake)

            loss_com = pre_loss(torch.mul(out_com, mask), torch.mul(data, mask)) + adv_loss(out, real_label)

            loss_com.backward()
            opt_com.step()


if __name__ == '__main__':
    # Some initial settings
    parser = argparse.ArgumentParser(description='Training of completion network for inpainting.')
    parser.add_argument('-batch_size', type=int, default=4,
                        help='Batch size (default = 128).')
    parser.add_argument('-num_workers', type=int, default=1,
                        help='Number of workers to load data (default = 2).')
    parser.add_argument('-epochs', type=int, default=10,
                        help='Number of epochs (default = 10).')
    parser.add_argument('-csv_dir', type=str, default='../data/locations',
                        help='Location where the .csv file is stored that holds all file names (default = ../data/locations).')
    parser.add_argument('-mask_dir', type=str, default='../data/mask.png',
                        help='Location where the mask is stored (default = ../data/mask.png).')

    args = parser.parse_args()

    # Load completion network and discriminator.
    com = model_completion.Completion().cuda()
    dis = model_completion.Discriminator().cuda()

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

    # Interpolation to resize input to 256 x 256 for global discriminator.
    x = np.linspace(-1, 1, 256)
    y = np.linspace(-1, 1, 256)

    xv, yv = np.meshgrid(x, y)

    grid = np.stack((xv, yv), axis=2)
    grid = np.expand_dims(grid, 0)
    grid = np.repeat(grid, args.batch_size, 0)

    grid = torch.from_numpy(grid).float()
    grid = Variable(grid).cuda()

    for epoch in range(args.epochs):
        train(epoch, 2)

        # Save results.
        torch.save(com.state_dict(), '../state_dict/com_epoch_{}.pth'.format(epoch + 1))
        torch.save(dis.state_dict(), '../state_dict/gl_dis.pth_epoch_{}.pth'.format(epoch + 1))


