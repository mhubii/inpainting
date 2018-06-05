import argparse
from model_dcgan import Generator, Discriminator
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import utils


def train():
    fixed_rand_input = Variable(torch.randn(args.batch_size, 100).view(-1, 100, 1, 1).cuda())

    for idx, data in enumerate(tqdm(data_loader)):
        data = Variable(data.cuda())

        # Train discriminator with fake and real data.
        label_real = Variable(torch.ones(args.batch_size).cuda())
        label_fake = Variable(torch.zeros(args.batch_size).cuda())

        optimizer_dis.zero_grad()

        output_dis = dis(data).squeeze()
        loss_dis_real = criterion(output_dis, label_real)

        rand_input = Variable(torch.randn(args.batch_size, 100).view(-1, 100, 1, 1).cuda())
        output_dis = dis(gen(rand_input)).squeeze()
        loss_dis_fake = criterion(output_dis, label_fake)

        loss_dis = loss_dis_real + loss_dis_fake
        loss_dis.backward()
        optimizer_dis.step()

        # Train generator.
        optimizer_gen.zero_grad()
        rand_input = Variable(torch.randn(args.batch_size, 100).view(-1, 100, 1, 1).cuda())
        output_dis = dis(gen(rand_input)).squeeze()
        loss_gen = criterion(output_dis, label_real)
        loss_gen.backward()
        optimizer_gen.step()

        # Save exemplary images.
        if idx % 100 == 0:
            print('Generator loss: {}'.format(loss_gen.data[0]))
            print('Discriminator loss: {}'.format(loss_dis.data[0]))

            real = data
            save_image(real.data, '../img/progress/real_snippet.png')

            fake = gen(fixed_rand_input)
            save_image(fake.data, '../img/progress/fake_snippet_epoch_{}.png'.format(epoch))


if __name__ == '__main__':
    # Some initial settings.
    parser = argparse.ArgumentParser(description='Training of DCGAN for inpainting.')
    parser.add_argument('-batch_size', type=int, default=128,
                        help='Batch size (default = 128).')
    parser.add_argument('-num_workers', type=int, default=2,
                        help='Number of workers to load data (default = 2).')
    parser.add_argument('-epochs', type=int, default=20,
                        help='Number of epochs (default = 10).')
    parser.add_argument('-csv_dir', type=str, default='../data/locations_snippets',
                        help='Location where the .csv file is stored that holds all file names (default = ../data/locations_snippets).')

    args = parser.parse_args()

    # Load model and initialize weights.
    gen = Generator(128).cuda()
    dis = Discriminator(128).cuda()

    gen.weight_init(mean=0.0, std=0.02)
    dis.weight_init(mean=0.0, std=0.02)

    # Set up data loader.
    data_set = utils.RadonSnippets(args.csv_dir,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=(0.5,), std=(0.5,))
                                   ]))

    data_loader = torch.utils.data.DataLoader(data_set,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              drop_last=True)

    # Optimizer and loss.
    optimizer_gen = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_dis = torch.optim.Adam(dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss().cuda()

    # Train.
    for epoch in range(args.epochs):
        train()

        # Save checkpoints.
        torch.save(gen.state_dict(), '../state_dict/gen_epoch_{}.pth'.format(epoch + 1))
        torch.save(dis.state_dict(), '../state_dict/dis_epoch_{}.pth'.format(epoch + 1))
