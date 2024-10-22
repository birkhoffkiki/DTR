import torch
import torch.nn as nn
import math


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(
            0, math.sqrt(2.0 / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels))
        )
        # nn.init.kaiming_uniform_(m.weight.data, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class UNetDownBlock(nn.Module):
    def __init__(self, in_size, out_size, padding=1, batch_norm=True):
        super(UNetDownBlock, self).__init__()

        mid_size = (in_size + out_size) // 2

        block = []

        # apply 3 conv steps. in->mid, mid->mid, mid->out (Conv+BN+LeakyRelu)
        block.append(nn.Conv2d(in_size, mid_size, kernel_size=3, padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(mid_size))
        block.append(nn.LeakyReLU(0.1, inplace=True))

        block.append(nn.Conv2d(mid_size, out_size, kernel_size=3, padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=False)

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x) + self.shortcut(x)
        out = self.lrelu(out)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, padding=1, batch_norm=True):
        super(UNetUpBlock, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False)
        )
        self.conv_block = UNetDownBlock(in_size, out_size, padding, batch_norm)

    def forward(self, x):
        up = self.up(x)
        out = self.conv_block(up)
        return out


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class SimpleConv(nn.Module):
    def __init__(self, in_size, padding=1, batch_norm=True):
        super(SimpleConv, self).__init__()
        block = []
        block.append(nn.Conv2d(in_size, in_size, kernel_size=3, padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(in_size))
        block.append(nn.LeakyReLU(0.1, inplace=True))

        block.append(
            nn.Conv2d(
                in_size, in_size * 2, kernel_size=3, stride=2, padding=int(padding)
            )
        )
        if batch_norm:
            block.append(nn.BatchNorm2d(in_size * 2))

        self.block = nn.Sequential(*block)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_size, in_size * 2, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(in_size * 2),
        )
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.block(x) + self.shortcut(x)
        out = self.lrelu(out)
        return out


# Generator: attention U-NET
class Generator(nn.Module):
    def __init__(
        self,
        n_channels=64,
        in_channels=1,
        out_channels=1,
        padding=1,
        batch_norm=True,
        pooling_mode="maxpool",
    ):
        super(Generator, self).__init__()

        assert pooling_mode in ("maxpool", "avgpool")

        if pooling_mode == "maxpool":
            self.Pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        if pooling_mode == "avgpool":
            self.Pooling = nn.AvgPool2d(kernel_size=2, stride=2)

        first_channel_num = n_channels
        self.Conv1 = UNetDownBlock(
            in_channels, first_channel_num, padding, batch_norm
        )  # 1->64
        self.Conv2 = UNetDownBlock(
            first_channel_num, first_channel_num * 2, padding, batch_norm
        )  # 64->128
        self.Conv3 = UNetDownBlock(
            first_channel_num * 2, first_channel_num * 4, padding, batch_norm
        )  # 128->256
        self.Conv4 = UNetDownBlock(
            first_channel_num * 4, first_channel_num * 8, padding, batch_norm
        )  # 256->512
        self.Conv5 = UNetDownBlock(
            first_channel_num * 8, first_channel_num * 16, padding, batch_norm
        )  # 512->1024

        self.Up5 = UNetUpBlock(
            first_channel_num * 16, first_channel_num * 8, padding, batch_norm
        )  # 1024->512
        self.Att5 = Attention_block(
            F_g=first_channel_num * 8,
            F_l=first_channel_num * 8,
            F_int=first_channel_num * 4,
        )  # 512->512

        self.Up4 = UNetUpBlock(
            first_channel_num * 16, first_channel_num * 4, padding, batch_norm
        )  # cat(512,512)->256
        self.Att4 = Attention_block(
            F_g=first_channel_num * 4,
            F_l=first_channel_num * 4,
            F_int=first_channel_num * 2,
        )  # 256->256

        self.Up3 = UNetUpBlock(
            first_channel_num * 8, first_channel_num * 2, padding, batch_norm
        )  # cat(256,256)->128
        self.Att3 = Attention_block(
            F_g=first_channel_num * 2,
            F_l=first_channel_num * 2,
            F_int=first_channel_num,
        )  # 128->128

        self.Up2 = UNetUpBlock(
            first_channel_num * 4, first_channel_num, padding, batch_norm
        )  # cat(128,128)->64
        self.Att2 = Attention_block(
            F_g=first_channel_num, F_l=first_channel_num, F_int=first_channel_num // 2
        )  # 64->64

        self.last_conv1 = UNetDownBlock(
            first_channel_num * 2, first_channel_num // 2, padding, batch_norm
        )  # cat(64,64)->32
        self.last_conv2 = nn.Conv2d(
            first_channel_num // 2, out_channels, kernel_size=1, stride=1, padding=0
        )  # 32->3

        self.last_activation = nn.Tanh()

    def forward(self, x):
        # encoding path, assuming input x in (in_ch,256,256), first channel num = 64;
        x1 = self.Conv1(x)  # (64,256,256)

        x2 = self.Pooling(x1)  # (64,128,128)
        x2 = self.Conv2(x2)  # (128,128,128)

        x3 = self.Pooling(x2)  # (128,64,64)
        x3 = self.Conv3(x3)  # (256,64,64)

        x4 = self.Pooling(x3)  # (256,32,32)
        x4 = self.Conv4(x4)  # (512,32,32)

        x5 = self.Pooling(x4)  # (512,16,16)
        x5 = self.Conv5(x5)  # (1024,16,16)

        # decoding + concat path
        d4 = self.Up5(x5)  # (512,32,32)
        g4 = self.Att5(
            g=d4, x=x4
        )  # attention gate, x is from skip connection in down path, g is output of previou up block.
        cat4 = torch.cat((g4, d4), dim=1)  # (1024,32,32)

        d3 = self.Up4(cat4)  # (256,64,64)
        g3 = self.Att4(g=d3, x=x3)  # (256,64,64)
        cat3 = torch.cat((g3, d3), dim=1)  # (512,64,64)

        d2 = self.Up3(cat3)  # (128,128,128)
        g2 = self.Att3(g=d2, x=x2)  # (128,128,128)
        cat2 = torch.cat((g2, d2), dim=1)  # (256,128,128)

        d1 = self.Up2(cat2)  # (64,256,256)
        g1 = self.Att2(g=d1, x=x1)  # (64,256,256)
        cat1 = torch.cat((g1, d1), dim=1)  # (128,256,256)

        out = self.last_conv1(cat1)  # (32,256,256)
        out = self.last_conv2(out)  # (3,256,256)

        out = self.last_activation(out)
        return out


# Discriminator: CNN
class Discriminator(nn.Module):
    def __init__(self, n_channels, in_channels=1, padding=1, batch_norm=True, n_blocks=5):
        super(Discriminator, self).__init__()

        initial_block = []
        initial_block.append(
            nn.Conv2d(
                in_channels, n_channels, kernel_size=3, padding=int(padding)
            )
        )
        if batch_norm:
            initial_block.append(nn.BatchNorm2d(n_channels))
        initial_block.append(nn.LeakyReLU(0.1, inplace=True))
        self.initial_block = nn.Sequential(*initial_block)

        conv_path = []
        in_size = n_channels
        for i in range(n_blocks):
            conv_path.append(SimpleConv(in_size, padding=1, batch_norm=True))
            in_size = in_size * 2
        self.conv_path = nn.Sequential(*conv_path)

        fc = []
        fc.append(nn.Linear(in_size, in_size // 2))
        fc.append(nn.LeakyReLU(0.1, inplace=True))
        fc.append(nn.Dropout(p=0.5))
        fc.append(nn.Linear(in_size // 2, 1))

        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        x = self.initial_block(x)
        x = self.conv_path(x)

        x = torch.mean(x, 3)
        x = torch.mean(x, 2)

        x = self.fc(x)

        return x
