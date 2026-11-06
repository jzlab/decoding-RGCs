import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

        self.time_proj = nn.Linear(time_dim, out_channels)

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):

        x = self.up(x)

        # Handle odd image sizes
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)

        x = nn.functional.pad(
            x,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
            ]
        )

        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        features=(64, 128, 256, 512),
    ):
        super().__init__()

        self.inc = DoubleConv(in_channels, features[0])

        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        self.bottleneck = DoubleConv(features[3], features[3] * 2)

        self.up3 = Up(features[3] * 2, features[3])
        self.up2 = Up(features[3], features[2])
        self.up1 = Up(features[2], features[1])
        self.up0 = Up(features[1], features[0])

        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, t, condition=None):

        # stack the noisy image (x) and the condition variable
        x = torch.cat([x, condition], dim=1)
        t = self.time_embed(t)

        # the encoding stage
        x1 = self.inc(x, t)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)

        # bottleneck
        x5 = self.bottleneck(x4, t)

        # decoding stage
        x = self.up3(x5, x4, t)
        x = self.up2(x, x3, t)
        x = self.up1(x, x2, t)
        x = self.up0(x, x1, t)

        return self.outc(x, t)