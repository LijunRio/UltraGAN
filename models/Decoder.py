from torch import nn
import torch
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1,bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes,
            out_planes,
            use_batchnorm =True):
    block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv3x3(in_planes, out_planes),
            nn.LeakyReLU(0.2, True))

    return block

class UpsampleBlock(nn.Module):
    def __init__(self,in_channels,up_scale):
        super(UpsampleBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels,in_channels* up_scale ** 2, kernel_size=3,padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.LeakyReLU(0.2, True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, x):
        out = self.block(x)
        return out+x

class ResBlockv2(nn.Module):
    def __init__(self, channels):
        super(ResBlockv2, self).__init__()
        self.conv1 = conv3x3(channels, channels)
        self.prelu = nn.LeakyReLU(0.2, True)
        self.conv2 = conv3x3(channels, channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.prelu(residual)
        residual = self.conv2(residual)

        return x + residual

class ResBlockv3(nn.Module):
    def __init__(self, channels):
        super(ResBlockv3, self).__init__()
        self.conv1 = conv3x3(channels, channels)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(channels, channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.prelu(residual)
        residual = self.conv2(residual)

        return x + residual

class RDN_conv(nn.Module):
    def __init__(self,inchanels,growth_rate):
        super(RDN_conv,self).__init__()
        self.conv1 = conv3x3(inchanels, growth_rate)
        self.relu = nn.PReLU()
    def forward(self,x):
        output = self.relu(self.conv(x))
        return torch.cat((x,output),1)

class RDN_block(nn.Module):
    def __init__(self, inchannels,C,growth_rate,):
        super(RDN_block, self).__init__()

        convs = []
        for i in range(C):
            convs.append(RDN_conv(inchannels + i * growth_rate, growth_rate))
        self.conv = nn.Sequential(*convs)
        # local_feature_fusion
        self.LFF = nn.Conv2d(inchannels + C * growth_rate, inchannels,kernel_size = 1,padding = 0,stride =1)
    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        # local residual learning
        return lff + x

class baseDecoder(nn.Module):
    def __init__(self,input_dim,feature_base_dim):
        super(baseDecoder, self).__init__()
        self.input_dim = input_dim
        self.feature_base_dim = feature_base_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM

        # -> ngf x 4 x 4
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.feature_base_dim * 4 * 4),
            nn.BatchNorm1d(self.feature_base_dim * 4 * 4),
            nn.LeakyReLU(0.2, True))

        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = upBlock(self.feature_base_dim, self.feature_base_dim // 2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = upBlock(self.feature_base_dim // 2, self.feature_base_dim // 4)
        # -> ngf/8 x 32 x 32
        self.upsample3 = upBlock(self.feature_base_dim // 4, self.feature_base_dim // 8)
        # -> ngf/16 x 64 x 64
        self.upsample4 = upBlock(self.feature_base_dim // 8, self.feature_base_dim // 16)
        # -> 1 x 64 x 64
        self.to_img = nn.Sequential(
            conv3x3(self.feature_base_dim // 16, 1),
            nn.Tanh()
        )
    def forward(self, input):
        h_code = self.fc(input)
        h_code = h_code.view(-1, self.feature_base_dim, 4, 4)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        # state size 1 x 64 x 64
        pre_img = self.to_img(h_code)
        return pre_img

class baseDecoderv2(nn.Module):
    def __init__(self,
                 input_dim,
                 feature_base_dim,
                 uprate):
        super(baseDecoderv2, self).__init__()
        self.input_dim = input_dim
        self.feature_base_dim = feature_base_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        self.uprate = uprate
        # -> ngf x 1 x 1
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.feature_base_dim),
            nn.BatchNorm1d(self.feature_base_dim),
            nn.LeakyReLU(0.2, True))
        self.upsamples = nn.Sequential()
        for i in range(self.uprate):
            # ngf//2**i x k x k -> ngf/2**(i+1) x k x k
            self.upsamples.add_module("upsample_{}".format(i),upBlock(self.feature_base_dim//2**i, self.feature_base_dim // 2**(i+1)))
            self.upsamples.add_module("res_{}".format(i),ResBlock(self.feature_base_dim // 2**(i+1)))
        # -> 1 x 64 x 64
        self.to_img = nn.Sequential(
            nn.Conv2d(self.feature_base_dim // 2**self.uprate, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )
    def forward(self, input):
        h_code = self.fc(input)
        h_code = h_code.view(-1, self.feature_base_dim, 1, 1)
        h_code = self.upsamples(h_code)
        # state size 1 x 32 x 32
        pre_img = self.to_img(h_code)
        return pre_img

class baseDecoderv3(nn.Module):
    def __init__(self,
                 input_dim,
                 feature_base_dim,
                 uprate):
        super(baseDecoderv3, self).__init__()
        self.input_dim = input_dim
        self.feature_base_dim = feature_base_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        self.uprate = uprate

        self.upsamples = nn.Sequential()
        for i in range(self.uprate):
            # ngf//2**i x k x k -> ngf/2**(i+1) x k x k
            self.upsamples.add_module("res_{}".format(i), ResBlockv2(self.feature_base_dim // 2 ** i))
            self.upsamples.add_module("upsample_{}".format(i),
                                      upBlock(self.feature_base_dim // 2 ** i, self.feature_base_dim // 2 ** (i + 1)))

        # -> 1 x 64 x 64
        self.to_img = nn.Sequential(
            nn.Conv2d(self.feature_base_dim // 2**self.uprate, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, input):
        h_code = input
        h_code = h_code.view(-1, self.feature_base_dim, 1, 1)
        h_code = self.upsamples(h_code)
        # state size 1 x 32 x 32
        pre_img = self.to_img(h_code)
        return pre_img


class PDecoderv2(nn.Module):
    def __init__(self,
                 input_dim,
                 feature_base_dim):
        super(PDecoderv2, self).__init__()
        self.input_dim = input_dim
        self.feature_base_dim = feature_base_dim
        self.num_res = 3

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.feature_base_dim//2),
            nn.LeakyReLU(0.2, True))

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.feature_base_dim//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True))

        res_layers = [ResBlockv2(self.feature_base_dim) for i in range(self.num_res)]
        self.Res_block = nn.Sequential(*res_layers)
        self.upblock = upBlock(self.feature_base_dim, self.feature_base_dim // 2)

        self.to_img = nn.Sequential(
            nn.Conv2d(self.feature_base_dim // 2, 1, kernel_size=1),
            nn.Tanh()
        )



    def forward(self, image, c_code):
        x = self.conv1(image)
        s_size = x.size(2)

        c_code = self.fc(c_code)
        c_code = c_code.view(-1, self.feature_base_dim//2, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)

        # state size (ngf+egf) x in_size x in_size
        h_c_code = torch.cat((c_code, x), 1)

        # state size ngf x in_size x in_size
        out_code = self.Res_block(h_c_code)
        out_code = self.upblock(out_code)
        # state size ngf/2 x 2in_size x 2in_size
        pre_img = self.to_img(out_code)

        return pre_img*0.5 + F.interpolate(image,scale_factor=2,mode="bilinear")*0.5

class PDecoderv3(nn.Module):
    def __init__(self,
                 input_dim,
                 feature_base_dim,
                 num_res = 4):
        super(PDecoderv3, self).__init__()
        self.input_dim = input_dim
        self.feature_base_dim = feature_base_dim
        self.num_res = num_res

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.feature_base_dim//2),
            nn.LeakyReLU(0.2, True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.feature_base_dim//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
        )

        res_layers = [ResBlockv3(self.feature_base_dim) for i in range(self.num_res)]
        self.Res_block = nn.Sequential(*res_layers)
        self.upblock = UpsampleBlock(self.feature_base_dim, 2)

        self.to_img = nn.Sequential(
            nn.Conv2d(self.feature_base_dim, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, image, c_code):
        x = self.conv1(image)
        s_size = x.size(2)

        c_code = self.fc(c_code)
        c_code = c_code.view(-1, self.feature_base_dim//2, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)

        # state size (ngf+egf) x in_size x in_size
        h_c_code = torch.cat((c_code, x), 1)

        # state size ngf x in_size x in_size
        res = self.Res_block(h_c_code)
        res += h_c_code
        # res = res + h_c_code
        out_code = self.upblock(res)
        # state size ngf/2 x 2in_size x 2in_size
        pre_img = self.to_img(out_code)

        return pre_img*0.5 + F.interpolate(image,scale_factor=2,mode="bilinear")*0.5


class PDecoder(nn.Module):
    def __init__(self, input_dim,feature_base_dim):
        super(PDecoder, self).__init__()
        self.input_dim = input_dim
        self.feature_base_dim = feature_base_dim
        self.num_res = 3
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.feature_base_dim//2),
            nn.BatchNorm1d(self.feature_base_dim//2),
            nn.LeakyReLU(0.2, True))

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.feature_base_dim//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True))

        res_layers = [ResBlock(self.feature_base_dim) for i in range(self.num_res)]
        self.Res_block = nn.Sequential(*res_layers)
        self.upblock = upBlock(self.feature_base_dim, self.feature_base_dim // 2)


        self.to_img = nn.Sequential(
            nn.Conv2d(self.feature_base_dim // 2, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, image, c_code):
        x = self.conv1(image)
        s_size = x.size(2)

        c_code = self.fc(c_code)
        c_code = c_code.view(-1, self.feature_base_dim//2, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)

        # state size (ngf+egf) x in_size x in_size
        h_c_code = torch.cat((c_code, x), 1)

        # state size ngf x in_size x in_size
        out_code = self.Res_block(h_c_code)
        out_code = self.upblock(out_code)
        # state size ngf/2 x 2in_size x 2in_size
        pre_img = self.to_img(out_code)

        return pre_img*0.5 + F.interpolate(image,scale_factor=2)*0.5

class MultiscaleDecoder(nn.Module):
    def __init__(self,
                 decoders):
        super(MultiscaleDecoder, self).__init__()

        self.num_layer = len(decoders)
        self.decoders = nn.ModuleList(decoders)

    def forward(self, txt_emded, layer_id):

        for i in range(layer_id + 1):
            decoder = self.decoders[i]
            # resize the final image to different size for the generator
            if i == 0:
                # The first generator only generate with text embedding
                pre_image = decoder(txt_emded)
            else:
                pre_image = decoder(pre_image, txt_emded)

        return pre_image
