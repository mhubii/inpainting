import argparse
from model import DCGAN
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import utils


def train():
    for data in data_loader:
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


if __name__ == '__main__':
    # Some initial settings.
    parser = argparse.ArgumentParser(description='Training of DCGAN for inpainting.')
    parser.add_argument('-batch_size', type=int, default=128,
                        help='Batch size (default = 128).')
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
                                              batch_size=args.batch_size)

    # Optimizer and loss.
    optimizer_gen = torch.optim.Adam(model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_dis = torch.optim.Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.MSELoss()

    # Train.
    for epoch in range(args.epochs):
        train()

    # Save results.
    torch.save(model.state_dict(), 'model_dcgan.pth')
