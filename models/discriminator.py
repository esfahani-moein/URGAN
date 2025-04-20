import torch
from torch import nn, Tensor
from typing import Any

class DiscriminatorForVGG(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
    ) -> None:
        super(DiscriminatorForVGG, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 128 x 128
            nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 64 x 64
            nn.Conv2d(channels, channels, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, int(2 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(2 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 32 x 32
            nn.Conv2d(int(2 * channels), int(2 * channels), (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(2 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(2 * channels), int(4 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(4 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 16 x 16
            nn.Conv2d(int(4 * channels), int(4 * channels), (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(4 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(4 * channels), int(8 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 8 x 8
            nn.Conv2d(int(8 * channels), int(8 * channels), (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(8 * channels), int(8 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 4 x 4
            nn.Conv2d(int(8 * channels), int(8 * channels), (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(int(8 * channels) * 4 * 4, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        return out


def discriminator_for_vgg(**kwargs) -> DiscriminatorForVGG:
    model = DiscriminatorForVGG(**kwargs)
    return model