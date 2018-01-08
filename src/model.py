import torch
import torch.nn as nn


class DCGAN(nn.Module):
    """
        Deep convolutional generative adversarial neural
        network according to the guidelines of Alec Radford,
        Luke Metz and Soumith Chintala:

         arXiv: https://arxiv.org/pdf/1511.06434.pdf

    """

    def __init__(self):
        super(DCGAN, self).__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, 3, 1),
            nn.ConvTranspose2d(1024, 512, 5, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 5, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, 5, 2, 1),
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(1, 128, 5, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 5, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 5, 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 5, 2),
        )

    def forward(self, x):
        # Reshape.
        out = x.view(-1, 100, 1, 1)

        # Pass forward through generator and discriminator.
        out = self.generator(out)
        out = self.discriminator(out)
        return out

    def forward_generator(self, x):
        # Reshape.
        out = x.view(-1, 100, 1, 1)

        # Pass forward through generator.
        out = self.generator(out)
        return out

    def forward_discriminator(self, x):
        # Pass forward through discriminator.
        out = self.discriminator(x)
        return out
