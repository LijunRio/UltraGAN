from torch import nn
import torch
import functools
import numpy as np
# from tools import SpectralNorm
from .tools import SpectralNorm



def Convblock(in_planes,
              out_planes,
              kernel=3,
              stride=1,
              padding=1):
    "convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride,
                     padding=padding, bias=False)


class ResBlock(nn.Module):
    def __init__(self,
                 channel,
                 norm):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            Convblock(channel, channel),
            norm(channel),
            nn.LeakyReLU(0.2, True),
            Convblock(channel, channel),
            norm(channel)
        )

    def forward(self, x):
        out = self.block(x)
        return out + x


def downBlock(in_planes,
              out_planes,
              kernel,
              padding,
              norm):
    block = nn.Sequential(
        Convblock(in_planes, out_planes, kernel=kernel, stride=2, padding=padding),
        norm(out_planes),
        nn.LeakyReLU(0.2, True))

    return block


def sameBlock(in_planes,
              out_planes,
              kernel,
              padding,
              norm):
    block = nn.Sequential(
        Convblock(in_planes, out_planes, kernel=kernel, stride=1, padding=padding),
        norm(out_planes),
        nn.LeakyReLU(0.2, True))

    return block


class SNDiscriminator(nn.Module):
    def __init__(self,
                 base_feature,
                 txt_input_dim,
                 down_rate,
                 norm="InstanceNorm"):
        super(SNDiscriminator, self).__init__()
        self.feature_base_dim = base_feature
        self.txt_input_dim = txt_input_dim
        self.down_rate = down_rate
        self.txt_dim = 128

        kw = 3
        padw = int(np.ceil((kw - 1.0) / 2))

        # 32 x 32 x 1 -> 32 x 32 x bf
        nf = self.feature_base_dim

        self.conv1SN = nn.Sequential(
            SpectralNorm(nn.Conv2d(1,
                                   nf,
                                   kernel_size=kw,
                                   stride=1,
                                   padding=padw)),
            nn.LeakyReLU(0.2, True))

        # 32 x 32 x bf -> 32 x 32 x bf
        self.downs = nn.Sequential()
        for i in range(self.down_rate):
            nf_pre = nf
            nf = min(nf * 2, 512)
            block = nn.Sequential(
                SpectralNorm(nn.Conv2d(nf_pre, nf, kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2, True))
            self.downs.add_module('down_{}'.format(i), block)

        # 1 x 1 x input_dim -> 1 x 1 x (8 x bf)
        self.text = nn.Sequential(SpectralNorm(nn.Linear(self.txt_input_dim, self.txt_dim)),
                                  nn.LeakyReLU(0.2, True))

        nf = nf + self.txt_dim
        self.output = nn.Sequential(
            nn.Conv2d(nf,
                      1,
                      kernel_size=4,
                      padding=0)
        )

    def forward(self, x, txt_embedding):
        # 64 x 64 x 1 -> 64 x 64 x bf
        x = self.conv1SN(x)

        x = self.downs(x)
        s_size = x.size(2)
        # 1 x 1 x input_dim -> 1 x 1 x (8 x bf)
        embedding = self.text(txt_embedding)

        # 1 x 1 x (8 x bf) -> 4 x 4 x (8 x bf)
        embedding = embedding.view(-1, self.txt_dim, 1, 1)
        embedding = embedding.repeat(1, 1, s_size, s_size)

        # 4 x 4 x (8 x bf)+ 4 x 4 x (8 x bf) -> 4 x 4 x (16 x bf)
        x = torch.cat((x, embedding), dim=1)

        # 4 x 4 x (16 x bf) -> 1
        return self.output(x)


class PDiscriminator(nn.Module):
    '''Patch discriminator, success in 256
    do not change'''

    def __init__(self,
                 base_feature,  # dis_channel_size = 64
                 txt_input_dim,  # D_channel_size = 512
                 down_rate,  # dr，传入参数 dr = self.base_ratio - 2 + layer_id
                 norm="InstanceNorm"):
        super(PDiscriminator, self).__init__()
        self.feature_base_dim = base_feature
        self.txt_input_dim = txt_input_dim
        self.down_rate = down_rate
        self.txt_dim = 128

        if norm == "InstanceNorm":
            self.norm2d = nn.InstanceNorm2d
        elif norm == "BatchNorm":
            self.norm2d = nn.BatchNorm2d

        kw = 3
        padw = int(np.floor((kw - 1.0) / 2))

        # 32 x 32 x 1 -> 32 x 32 x bf
        nf = self.feature_base_dim
        self.conv1 = nn.Sequential(
            Convblock(1,
                      nf,
                      kernel=kw,
                      stride=1,
                      padding=padw),
            nn.LeakyReLU(0.2, True))

        # 32 x 32 x bf -> 32 x 32 x bf
        self.downs = nn.Sequential()
        for i in range(self.down_rate):
            nf_pre = nf
            nf = min(nf * 2, 512)
            block = nn.Sequential(
                Convblock(nf_pre, nf, kernel=kw, stride=2, padding=padw),
                self.norm2d(nf),
                nn.LeakyReLU(0.2, True)
            )
            self.downs.add_module('down_{}'.format(i), block)

        # 1 x 1 x input_dim -> 1 x 1 x (8 x bf)
        self.textfc = nn.Linear(self.txt_input_dim, self.txt_dim)
        self.textBN = self.norm2d(self.txt_dim)
        self.textAcc = nn.LeakyReLU(0.2, True)

        nf = nf + self.txt_dim
        self.output = nn.Sequential(
            nn.Conv2d(nf,
                      1,
                      kernel_size=4,
                      padding=1)
        )

    def forward(self, x, txt_embedding):
        # 64 x 64 x 1 -> 64 x 64 x bf
        x = self.conv1(x)

        x = self.downs(x)
        s_size = x.size(2)
        # 1 x 1 x input_dim -> 1 x 1 x (8 x bf)
        embedding = self.textfc(txt_embedding)

        # 1 x 1 x (8 x bf) -> 4 x 4 x (8 x bf)
        embedding = embedding.view(-1, self.txt_dim, 1, 1)
        embedding = embedding.repeat(1, 1, s_size, s_size)
        embedding = self.textBN(embedding)
        embedding = self.textAcc(embedding)

        # 4 x 4 x (8 x bf)+ 4 x 4 x (8 x bf) -> 4 x 4 x (16 x bf)
        x = torch.cat((x, embedding), dim=1)

        # 4 x 4 x (16 x bf) -> 1
        return self.output(x)


class baseDiscriminator(nn.Module):
    def __init__(self,
                 base_feature,
                 txt_input_dim,
                 down_rate,
                 norm="InstanceNorm"):
        super(baseDiscriminator, self).__init__()
        self.feature_base_dim = base_feature
        self.txt_input_dim = txt_input_dim
        self.down_rate = down_rate
        self.txt_dim = 128

        if norm == "InstanceNorm":
            self.norm2d = nn.InstanceNorm2d
        elif norm == "BatchNorm":
            self.norm2d = nn.BatchNorm2d

        kw = 3
        padw = int(np.floor((kw - 1.0) / 2))

        # 32 x 32 x 1 -> 32 x 32 x bf
        nf = self.feature_base_dim
        self.conv1 = nn.Sequential(
            Convblock(1,
                      nf,
                      kernel=kw,
                      stride=1,
                      padding=padw),
            nn.LeakyReLU(0.2, True))

        # 32 x 32 x bf -> 32 x 32 x bf
        self.downs = nn.Sequential()
        for i in range(self.down_rate):
            nf_pre = nf
            nf = min(nf * 2, 512)
            self.downs.add_module('down_{}'.format(i), downBlock(in_planes=nf_pre,
                                                                 out_planes=nf,
                                                                 kernel=kw,
                                                                 padding=padw,
                                                                 norm=self.norm2d))

        # 1 x 1 x input_dim -> 1 x 1 x (8 x bf)
        self.textfc = nn.Linear(self.txt_input_dim, self.txt_dim)
        self.textBN = self.norm2d(self.txt_dim)
        self.textAcc = nn.LeakyReLU(0.2, True)

        nf = nf + self.txt_dim
        self.output = nn.Sequential(
            nn.Conv2d(nf,
                      1,
                      kernel_size=4,
                      padding=0)
        )

    def forward(self, x, txt_embedding):
        # 64 x 64 x 1 -> 64 x 64 x bf
        x = self.conv1(x)

        x = self.downs(x)
        s_size = x.size(2)
        # 1 x 1 x input_dim -> 1 x 1 x (8 x bf)
        embedding = self.textfc(txt_embedding)

        # 1 x 1 x (8 x bf) -> 4 x 4 x (8 x bf)
        embedding = embedding.view(-1, self.txt_dim, 1, 1)
        embedding = embedding.repeat(1, 1, s_size, s_size)
        embedding = self.textBN(embedding)
        embedding = self.textAcc(embedding)

        # 4 x 4 x (8 x bf)+ 4 x 4 x (8 x bf) -> 4 x 4 x (16 x bf)
        x = torch.cat((x, embedding), dim=1)

        # 4 x 4 x (16 x bf) -> 1
        return self.output(x)


class ResDiscriminator(nn.Module):
    def __init__(self,
                 base_feature,
                 txt_input_dim,
                 down_rate,
                 norm="InstanceNorm"):
        super(ResDiscriminator, self).__init__()
        self.feature_base_dim = base_feature
        self.txt_input_dim = txt_input_dim
        self.down_rate = down_rate
        self.txt_dim = 128

        if norm == "InstanceNorm":
            self.norm2d = nn.InstanceNorm2d
        elif norm == "BatchNorm":
            self.norm2d = nn.BatchNorm2d

        kw = 3
        padw = int(np.ceil((kw - 1.0) / 2))

        # 32 x 32 x 1 -> 32 x 32 x bf
        nf = self.feature_base_dim
        self.conv1 = nn.Sequential(
            Convblock(1,
                      nf,
                      kernel=kw,
                      stride=1,
                      padding=padw),
            nn.LeakyReLU(0.2, True))

        # 32 x 32 x bf -> 32 x 32 x bf
        self.downs = nn.Sequential()
        for i in range(self.down_rate):
            nf_pre = nf
            nf = min(nf * 2, 512)
            self.downs.add_module('Rse_{}'.format(i), ResBlock(channel=nf_pre,
                                                               norm=self.norm2d))
            self.downs.add_module('down_{}'.format(i), downBlock(in_planes=nf_pre,
                                                                 out_planes=nf,
                                                                 kernel=kw,
                                                                 padding=padw,
                                                                 norm=self.norm2d))

        # 1 x 1 x input_dim -> 1 x 1 x (8 x bf)
        self.textfc = nn.Linear(self.txt_input_dim, self.txt_dim)
        self.textnorm = self.norm2d(self.txt_dim)
        self.textAcc = nn.LeakyReLU(0.2, True)

        nf = nf + self.txt_dim
        self.output = nn.Sequential(
            nn.Conv2d(nf,
                      1,
                      kernel_size=4,
                      padding=1)
        )

    def forward(self, x, txt_embedding):
        # 64 x 64 x 1 -> 64 x 64 x bf
        x = self.conv1(x)

        x = self.downs(x)
        s_size = x.size(2)
        # 1 x 1 x input_dim -> 1 x 1 x (8 x bf)
        embedding = self.textfc(txt_embedding)

        # 1 x 1 x (8 x bf) -> 4 x 4 x (8 x bf)
        embedding = embedding.view(-1, self.txt_dim, 1, 1)
        embedding = embedding.repeat(1, 1, s_size, s_size)
        embedding = self.textnorm(embedding)
        embedding = self.textAcc(embedding)

        # 4 x 4 x (8 x bf)+ 4 x 4 x (8 x bf) -> 4 x 4 x (16 x bf)
        x = torch.cat((x, embedding), dim=1)

        # 4 x 4 x (16 x bf) -> 1
        return self.output(x)


class MSDiscriminator(nn.Module):
    def __init__(self,
                 base_feature,
                 txt_input_dim,
                 down_rate,
                 num_stage,
                 use_batchnorm=True):
        super(MSDiscriminator, self).__init__()
        self.feature_base_dim = base_feature
        self.txt_input_dim = txt_input_dim
        self.down_rate = down_rate + num_stage
        self.num_stage = num_stage

        self.from_image = nn.ModuleList([nn.Sequential(Convblock(in_planes=1,
                                                                 out_planes=self.feature_base_dim * 2 ** max(0, i - 1)),
                                                       nn.LeakyReLU(0.2)) for i in range(num_stage)])

        self.downs_image = []
        for i in range(self.num_stage):
            self.downs_image.append(downBlock(self.feature_base_dim * 2 ** i,
                                              self.feature_base_dim * 2 ** i,
                                              use_batchnorm=use_batchnorm))
        self.downs_image = nn.ModuleList(self.downs_image)

        self.downs = nn.Sequential()
        for i in range(self.num_stage - 1, self.down_rate - 1):
            self.downs.add_module('down_{}'.format(i), downBlock(self.feature_base_dim * 2 ** i,
                                                                 self.feature_base_dim * 2 ** (i + 1),
                                                                 use_batchnorm=use_batchnorm))

        self.textRepeat = nn.Sequential(
            nn.Linear(self.txt_input_dim, self.feature_base_dim * 2 ** (self.down_rate - 1)),
            nn.BatchNorm1d(self.feature_base_dim * 2 ** (self.down_rate - 1)),
            nn.ReLU(),
            # nn.ReLU(inplace=True),
        )

        self.output = nn.Sequential(
            nn.Conv2d(self.feature_base_dim * 2 ** (self.down_rate), 1, kernel_size=4)
        )

    def forward(self, images, txt_embedding):
        images = images[::-1]
        # 128 x 128 x 1 -> 128 x 128 x bf
        x = self.from_image[0](images[0])
        # 128 x 128 x bf -> 64 x 64 x bf
        y = self.downs_image[0](x)

        for i in range(1, self.num_stage):
            # 64 x 64 x 1 -> 64 x 64 x bf
            # 32 x 32 x 1 -> 32 x 32 x 2*bf
            x = self.from_image[i](images[i])
            # 64 x 64 x bf + 64 x 64 x bf = 64 x 64 x 2*bf
            # 32 x 32 x 2*bf + 32 x 32 x 2*bf = 32 x 32 x 4*bf
            x = torch.cat((x, y), dim=1)
            # 64 x 64 x 2*bf -> 32 x 32 x 2*bf
            # 32 x 32 x 4*bf -> 16 x 16 x 4*bf
            x = self.downs_image[i](x)
            y = x

        # 16 x 16 x 4*bf -> 8 x 8 x 8*bf
        # 8 x 8 x 8*bf -> 4 x 4 x 16*bf
        x = self.downs(x)

        s_size = x.size(2)
        # 1 x 1 x input_dim -> 1 x 1 x (16 x bf)
        embedding = self.textRepeat(txt_embedding)

        # 1 x 1 x (16 x bf) -> 4 x 4 x (16 x bf)
        embedding = embedding.view(-1, self.feature_base_dim * 2 ** (self.down_rate - 1), 1, 1)
        embedding = embedding.repeat(1, 1, s_size, s_size)

        # 4 x 4 x (16 x bf)+ 4 x 4 x (16 x bf) -> 4 x 4 x (32 x bf)
        x = torch.cat((x, embedding), dim=1)

        # 4 x 4 x (32 x bf) -> 1
        return self.output(x)


class noCon_Discriminator(nn.Module):
    def __init__(self,
                 base_feature,
                 txt_input_dim,
                 down_rate,
                 norm="InstanceNorm"):
        super(noCon_Discriminator, self).__init__()
        self.feature_base_dim = base_feature
        self.txt_input_dim = txt_input_dim
        self.down_rate = down_rate
        self.txt_dim = 128

        if norm == "InstanceNorm":
            self.norm2d = nn.InstanceNorm2d
        else:
            self.norm2d = nn.BatchNorm2d

        kw = 3
        padw = int(np.ceil((kw - 1.0) / 2))

        # 32 x 32 x 1 -> 32 x 32 x bf
        nf = self.feature_base_dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      nf,
                      kernel_size=kw,
                      stride=1,
                      padding=padw),
            nn.LeakyReLU(0.2, True))

        # 32 x 32 x bf -> 32 x 32 x bf
        self.downs = nn.Sequential()
        for i in range(self.down_rate):
            nf_pre = nf
            nf = min(nf * 2, 512)
            self.downs.add_module('down_{}'.format(i), downBlock(in_planes=nf_pre,
                                                                 out_planes=nf,
                                                                 kernel=kw,
                                                                 padding=padw,
                                                                 norm=self.norm2d))

        self.output = nn.Sequential(
            nn.Conv2d(nf,
                      1,
                      kernel_size=kw,
                      padding=padw)
        )

    def forward(self, x, txt_embedding):
        # 64 x 64 x 1 -> 64 x 64 x bf
        x = self.conv1(x)

        x = self.downs(x)

        return self.output(x)


class PatchDiscriminator(nn.Module):
    def __init__(self,
                 base_feature,
                 txt_input_dim,
                 down_rate,
                 norm="InstanceNorm"):
        super(PatchDiscriminator, self).__init__()
        self.feature_base_dim = base_feature
        self.txt_input_dim = txt_input_dim
        self.down_rate = down_rate
        self.txt_dim = 128

        if norm == "InstanceNorm":
            self.norm2d = nn.InstanceNorm2d
        elif norm == "BatchNorm":
            self.norm2d = nn.BatchNorm2d

        kw = 3
        padw = int(np.ceil((kw - 1.0) / 2))

        # 32 x 32 x 1 -> 32 x 32 x bf
        nf = self.feature_base_dim
        self.conv1 = nn.Sequential(
            Convblock(1,
                      nf,
                      kernel=kw,
                      stride=2,
                      padding=padw),
            nn.LeakyReLU(0.2, True))

        # 32 x 32 x bf -> 32 x 32 x bf
        self.downs = nn.Sequential()
        for i in range(1, self.down_rate):
            nf_pre = nf
            nf = min(nf * 2, 512)
            self.downs.add_module('down_{}'.format(i), downBlock(in_planes=nf_pre,
                                                                 out_planes=nf,
                                                                 kernel=kw,
                                                                 padding=padw,
                                                                 norm=self.norm2d))

        # 1 x 1 x input_dim -> 1 x 1 x (8 x bf)
        self.textfc = nn.Linear(self.txt_input_dim, self.txt_dim)
        self.textnorm = self.norm2d(self.txt_dim)
        self.textAcc = nn.LeakyReLU(0.2, True)

        nf = nf + self.txt_dim

        self.output = nn.Sequential(
            nn.Conv2d(nf,
                      1,
                      kernel_size=4)
        )

    def forward(self, x, txt_embedding):
        # 64 x 64 x 1 -> 64 x 64 x bf
        x = self.conv1(x)

        x = self.downs(x)
        s_size = x.size(2)
        # 1 x 1 x input_dim -> 1 x 1 x (8 x bf)
        embedding = self.textfc(txt_embedding)

        # 1 x 1 x (8 x bf) -> 4 x 4 x (8 x bf)
        embedding = embedding.view(-1, self.txt_dim, 1, 1)
        embedding = embedding.repeat(1, 1, s_size, s_size)
        embedding = self.textnorm(embedding)
        embedding = self.textAcc(embedding)

        # 4 x 4 x (8 x bf)+ 4 x 4 x (8 x bf) -> 4 x 4 x (16 x bf)
        x = torch.cat((x, embedding), dim=1)

        # 4 x 4 x (16 x bf) -> 1
        return self.output(x)
