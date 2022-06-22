from torch import nn
import torch
from torch.nn import Parameter
import numpy as np


torch.autograd.set_detect_anomaly = True
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def MScal_gradient_penalty(netD, real_data, fake_data, txt_emded, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesvs = real_data
        elif type == 'fake':
            interpolatesvs = fake_data
        elif type == 'mixed':
            interpolatesvs = []
            for i in range(len(real_data)):
                alpha = torch.rand(real_data[i].shape[0], 1, device=device)
                alpha = alpha.expand(real_data[i].shape[0],
                                     real_data[i].nelement() // real_data[i].shape[0]).contiguous().view(
                    *real_data[i].shape)
                interpolatesv = alpha * real_data[i] + ((1 - alpha) * fake_data[i])
                interpolatesv.requires_grad_(True)
                interpolatesvs.append(interpolatesv)
        else:
            raise NotImplementedError('{} not implemented'.format(type))

        disc_interpolates = netD(interpolatesvs, txt_emded)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesvs,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data[0].size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

def cal_gradient_penalty(netD, real_data, fake_data, txt_emded, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':

            alpha = torch.rand(real_data.shape[0], 1, device=device)  # [96, 1]
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)  # [96, 1, 32, 32]
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)  # 真实的数据和假的数据，插值？
            interpolatesv.requires_grad_(True)

        else:
            raise NotImplementedError('{} not implemented'.format(type))

        disc_interpolates = netD(interpolatesv, txt_emded)
        # disc_interpolates = netD(interpolatesv.detach(), txt_emded.detach())
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)

        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

# def cal_gradient_penalty(netD, real_data, fake_data, txt_emded, device, type='mixed', constant=1.0, lambda_gp=10.0):
#     """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
#     Arguments:
#         netD (network)              -- discriminator network
#         real_data (tensor array)    -- real images
#         fake_data (tensor array)    -- generated images from the generator
#         device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
#         type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
#         constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
#         lambda_gp (float)           -- weight for this loss
#     Returns the gradient penalty loss
#     """
#     if lambda_gp > 0.0:
#         if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
#             interpolatesv = real_data
#         elif type == 'fake':
#             interpolatesv = fake_data
#         elif type == 'mixed':
#
#             alpha = torch.rand(real_data.shape[0], 1, device=device)
#             alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
#                 *real_data.shape)
#             interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
#             interpolatesv.requires_grad_(True)
#
#         else:
#             raise NotImplementedError('{} not implemented'.format(type))
#
#         disc_interpolates = netD(interpolatesv, txt_emded)
#         gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
#                                         grad_outputs=torch.ones(disc_interpolates.size()).to(device),
#                                         create_graph=True, retain_graph=True, only_inputs=True)
#         gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
#         gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
#         return gradient_penalty, gradients
#     else:
#         return 0.0, None


from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
            # loss = loss + self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def init_weights(m):
    # Initialization
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        # get the number of the inputs
        y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0, 1 / np.sqrt(y))


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
