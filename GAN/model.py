import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels=100, out_channels=3):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # (512 * 4 * 4)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(268),
            nn.ReLU(True),
            # (256 * 8 * 8)
            nn.ConvTranspose2d(256, 128, 4, 2,),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # (128 * 16 * 16)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # (64 * 32 * 32)
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # (3 * 64 * 64)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # input is (in_channels * 64 * 64)
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (64 * 32 * 32)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # (128 * 16 * 16)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # (256 * 8 * 8)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # (512 * 4 * 4)
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() # binary classifier
        )

    def forward(self, x):
        return self.model(x)
