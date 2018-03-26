import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Adversarial network for globally and locally consistent 
    image completion according to Satoshi Iizuka, 
    Edgar Simo-Serra and Hiroschi Ishikawa:

        Paper: http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf

"""


class Completion(nn.Module):

    def __init__(self):
        super(Completion, self).__init__()
        self.completion = nn.Sequential(nn.Conv2d(1, 64, 5, 1, 2),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 128, 3, 2, 1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 128, 3, 1, 1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 256, 3, 2, 1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, 3, 1, 1, 1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, 3, 1, 1, 1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, 3, 1, 2, 2),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, 3, 1, 4, 4),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, 3, 1, 8, 8),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, 3, 1, 16, 16),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, 3, 1, 1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, 3, 1, 1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(256, 128, 4, 2, 1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 128, 3, 1, 1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(128, 64, 4, 2, 1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 32, 3, 1, 1),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 1, 3, 1, 1),
                                        nn.Sigmoid())

    def forward(self, input):
        output = self.completion(input)

        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.local_discriminator = nn.Sequential(nn.Conv2d(1, 64, 5, 2),
                                                 nn.BatchNorm2d(64),
                                                 nn.ReLU(),
                                                 nn.Conv2d(64, 128, 5, 2),
                                                 nn.BatchNorm2d(128),
                                                 nn.ReLU(),
                                                 nn.Conv2d(128, 256, 5, 2),
                                                 nn.BatchNorm2d(256),
                                                 nn.ReLU(),
                                                 nn.Conv2d(256, 512, 5, 2),
                                                 nn.BatchNorm2d(512),
                                                 nn.ReLU(),
                                                 nn.Conv2d(512, 512, 5, 2),
                                                 nn.BatchNorm2d(512),
                                                 nn.ReLU())

        self.global_discriminator = nn.Sequential(nn.Conv2d(1, 64, 5, 2),
                                                  nn.BatchNorm2d(64),
                                                  nn.ReLU(),
                                                  nn.Conv2d(64, 128, 5, 2),
                                                  nn.BatchNorm2d(128),
                                                  nn.ReLU(),
                                                  nn.Conv2d(128, 256, 5, 2),
                                                  nn.BatchNorm2d(256),
                                                  nn.ReLU(),
                                                  nn.Conv2d(256, 512, 5, 2),
                                                  nn.BatchNorm2d(512),
                                                  nn.ReLU(),
                                                  nn.Conv2d(512, 512, 5, 2),
                                                  nn.BatchNorm2d(512),
                                                  nn.ReLU(),
                                                  nn.Conv2d(512, 512, 5, 2),
                                                  nn.BatchNorm2d(512),
                                                  nn.ReLU())
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(2048, 1)

    def forward(self, local_input, global_input):
        # Forward local and global discriminator.
        local_output = self.local_discriminator(local_input)
        local_output = local_output.view(local_output.shape[0], 1, 1, -1)
        local_output = self.fc1(local_output)

        global_output = self.global_discriminator(global_input)
        global_output = global_output.view(local_output.shape[0], 1, 1, -1)
        global_output = self.fc2(global_output)

        # Concatenate outputs and determine viability.
        output = torch.cat((local_output, global_output), 3)
        output = F.sigmoid(self.fc3(output))

        return output
