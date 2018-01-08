import argparse
from model import DCGAN
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from torchvision.utils import save_image
import utils


def train():
    fixed_rand_input = Variable(torch.rand(args.batch_size, 100).normal_(0, 1).cuda())

    for idx, data in enumerate(data_loader):
        data = Variable(data.float().cuda())
        data = torch.unsqueeze(data, 1)

        # Train discriminator with fake and real data.
        model.generator.eval()
        model.discriminator.train()

        label_real = Variable(torch.ones(args.batch_size).cuda())
        label_fake = Variable(torch.zeros(args.batch_size).cuda())

        optimizer_dis.zero_grad()

        output_dis_real = model.forward_discriminator(data)
        rand_input = Variable(torch.rand(args.batch_size, 100).cuda())
        output_dis_fake = model.forward(rand_input)

        loss_dis_real = criterion(output_dis_real, label_real)
        loss_dis_fake = criterion(output_dis_fake, label_fake)

        loss_dis = loss_dis_real + loss_dis_fake
        loss_dis.backward()
        optimizer_dis.step()

        # Train generator.
        model.generator.train()
        model.discriminator.eval()

        optimizer_gen.zero_grad()
        rand_input = Variable(torch.rand(args.batch_size, 100).cuda())
        output_gen = model.forward_generator(rand_input)
        loss_gen = criterion(output_gen, data)
        loss_gen.backward()
        optimizer_gen.step()

        # Save exemplary images.
        if idx % 1000 == 0:
            fake = model.forward_generator(fixed_rand_input)
            save_image(fake.data, '../img/progress/fake_snippet_epoch_{}.png'.format(epoch))


if __name__ == '__main__':
    # Some initial settings.
    parser = argparse.ArgumentParser(description='Training of DCGAN for inpainting.')
    parser.add_argument('-batch_size', type=int, default=128,
                        help='Batch size (default = 128).')
    parser.add_argument('-num_workers', type=int, default=2,
                        help='Number of workers to load data (default = 2).')
    parser.add_argument('-epochs', type=int, default=10,
                        help='Number of epochs (default = 10).')
    parser.add_argument('-csv_dir', type=str, default='../data/locations',
                        help='Location where the .csv file is stored that holds all file names (default = ../data/locations).')

    args = parser.parse_args()

    # Load model and initialize weights.
    model = DCGAN().cuda()
    model.apply(utils.weight_init)

    # Set up data loader.
    data_set = utils.RadonSnippets(args.csv_dir)
    data_loader = torch.utils.data.DataLoader(data_set,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers)

    # Optimizer and loss.
    optimizer_gen = torch.optim.Adam(model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_dis = torch.optim.Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.MSELoss().cuda()

    # Train.
    for epoch in range(args.epochs):
        train()

        # Save checkpoints.
        torch.save(model.state_dict(), '../state_dict/model_dcgan_epoch_{}.path'.format(epoch))
