import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models.Decoder import baseDecoderv3, PDecoderv3, MultiscaleDecoder
from models.Descriminator import PDiscriminator
from models.Encoder import HAttnEncoder
from utils.dataset import MyDataset2
from utils.processing import Rescale, ToTensor, Equalize, deNorm, get_time
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from tqdm import tqdm
from config import config as args
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models.tools import cal_gradient_penalty
import os
from utils.SSIM import *
import math


class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.device, self.device_ids = self._prepare_device(args.n_gpu)
        self.exp_name = args.exp_name
        self.dict_pth = args.vocab_path
        self.encoder_checkpoint = args.encoder_checkpoint
        self.decoder_checkpoint = args.decoder_checkpoint
        self.D_checkpoint = args.D_checkpoint
        self.check_create_checkpoint()

        word_dict = json.loads(open(self.dict_pth, 'r', encoding="utf_8_sig").read())
        self.vocab = word_dict[0]
        self.max_finding_len = word_dict[3]
        self.max_impression_len = word_dict[2]
        self.batch_size = args.batch_size
        self.beta1 = self.args.beta1
        self.DISP_FREQs = [10, 20, 30, 50]
        self.lambda_gp = [10.0, 10.0, 10.0, 10.0]
        self.each_step_epoch = self.args.each_step_epoch
        self.G_step = args.d_step
        self.D_step = args.g_step
        self.image_size = args.image_size
        content_losses = {"L2": nn.MSELoss(),
                          "L1": nn.L1Loss()}
        self.G_criterion = content_losses[args.content_loss].to(self.device)
        self.adv_loss_ratio = args.adv_loss_ratio
        self.pix_loss_ratio = args.pix_loss_ratio
        print(type(self.image_size))
        self.base_size = args.base_size
        self.encoder_resume = args.resume_encoder
        self.decoder_resume = args.resume_decoder

        self.trainset = MyDataset2(args, split='train', transform=transforms.Compose([
            Rescale(self.image_size),
            Equalize(),
            ToTensor()]))
        self.valset = MyDataset2(args, split='val', transform=transforms.Compose([
            Rescale(self.image_size),
            Equalize(),
            ToTensor()]))
        self.testset = MyDataset2(args, split='train', transform=transforms.Compose([
            Rescale(self.image_size),
            Equalize(),
            ToTensor()]))
        self.writer = SummaryWriter(os.path.join("runs", self.exp_name))

        self.base_ratio = int(np.log2(self.base_size))
        self.P_ratio = int(np.log2(self.image_size[0] // self.base_size))
        self.define_nets()
        self.decoder = nn.DataParallel(self.decoder, device_ids=self.device_ids)
        self.encoder = nn.DataParallel(self.encoder, device_ids=self.device_ids)
        self.load_model()

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".
                    format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def define_nets(self):
        """
        Encoder: text to features
        """
        self.encoder = HAttnEncoder(vocab_size=len(self.vocab),
                                    embed_size=self.args.embed_size,
                                    hidden_size=self.args.hidden_size,
                                    max_len=[self.max_finding_len,
                                             self.max_impression_len],
                                    unit=self.args.rnn_cell,
                                    feature_base_dim=self.args.d_channel_size
                                    ).to(self.device)

        """
        Decoder = BaseDecoder + iterator_add(Decoders)
        """
        decoders_list = []
        first_decoder = baseDecoderv3(input_dim=self.args.d_channel_size,
                                      feature_base_dim=self.args.d_channel_size,
                                      uprate=self.base_ratio).to(self.device)
        decoders_list.append(first_decoder)
        for i in range(1, self.P_ratio + 1):
            nf = 128
            pdecoder = PDecoderv3(input_dim=self.args.d_channel_size,
                                  feature_base_dim=nf).to(self.device)
            decoders_list.append(pdecoder)
        self.decoder = MultiscaleDecoder(decoders_list).to(self.device)

    def define_Discriminator(self, layer_id):
        '''Initialize a series of Discriminator'''
        discriminator_rate = self.base_ratio - 2 + layer_id
        self.D = PDiscriminator(base_feature=self.args.d_channel_size,
                                txt_input_dim=self.args.d_channel_size,
                                down_rate=discriminator_rate).to(self.device)
        if len(self.device_ids) > 1:
            self.D = nn.DataParallel(self.D, device_ids=self.device_ids)

    def define_dataloader(self, layer_id):
        self.train_dataloader = DataLoader(self.trainset,
                                           batch_size=self.batch_size[layer_id],
                                           shuffle=True,
                                           drop_last=False)
        self.val_dataloader = DataLoader(self.valset,
                                         batch_size=self.batch_size[layer_id],
                                         shuffle=True,
                                         drop_last=False)
        self.test_dataloader = DataLoader(self.testset,
                                          batch_size=self.batch_size[layer_id],
                                          shuffle=False,
                                          drop_last=True)

    def define_opt(self, layer_id):
        '''Define optimizer'''
        self.G_optimizer = torch.optim.Adam([{'params': self.encoder.parameters()}] +
                                            [{'params': self.decoder.parameters()}],
                                            lr=self.args.G_LR[layer_id], betas=(self.beta1, 0.999))
        # MultiStepLR:按照间隔调整学习率
        self.G_lr_scheduler = MultiStepLR(self.G_optimizer, milestones=self.args.lr_decay_epoch[layer_id], gamma=0.2)

        self.D_optimizer = torch.optim.Adam([{'params': self.D.parameters()}],
                                            lr=self.args.D_LR[layer_id], betas=(self.beta1, 0.999))
        self.D_lr_scheduler = MultiStepLR(self.D_optimizer, milestones=self.args.lr_decay_epoch[layer_id], gamma=0.2)

    def Loss_on_layer(self, image, finding, impression, layer_id, decoder):
        txt_emded, hidden = self.encoder(finding, impression)
        r_image = F.interpolate(image, size=(2 ** layer_id) * self.base_size)

        self.G_optimizer.zero_grad()
        pre_image = decoder(txt_emded, layer_id)
        loss = self.G_criterion(pre_image.float(), r_image.float())
        loss.backward()  # 反向传播
        self.G_optimizer.step()  # 更行生成器的所有参数
        return loss, pre_image, r_image

    def check_create_checkpoint(self):
        '''Check for the checkpoint path exists or not
        If not exist, create folder'''
        if os.path.exists(self.encoder_checkpoint) == False:
            os.makedirs(self.encoder_checkpoint)
        if os.path.exists(self.decoder_checkpoint) == False:
            os.makedirs(self.decoder_checkpoint)
        if os.path.exists(self.D_checkpoint) == False:
            os.makedirs(self.D_checkpoint)

    def load_model(self):
        if os.path.exists(self.encoder_resume):
            print("load checkpoint {}".format(self.encoder_resume))
            self.encoder.load_state_dict(torch.load(self.encoder_resume))
        else:
            print("checkpoint do not exists {}".format(self.encoder_resume))
        if os.path.exists(self.decoder_resume):
            print("load checkpoint {}".format(self.decoder_resume))
            self.decoder.load_state_dict(torch.load(self.decoder_resume))
        else:
            print("checkpoint do not exists {}".format(self.decoder_resume))

    def save_model(self, layer_id):
        torch.save(self.encoder.state_dict(), os.path.join(self.encoder_checkpoint,
                                                           "Encoder_Layer_{}_Time_{}_checkpoint.pth".format(layer_id,
                                                                                                            get_time())))
        torch.save(self.decoder.state_dict(), os.path.join(self.decoder_checkpoint,
                                                           "Decoder_Layer_{}_Time_{}_checkpoint.pth".format(layer_id,
                                                                                                            get_time())))
    def save_model2(self, layer_id):
        torch.save(self.encoder.state_dict(), os.path.join(self.encoder_checkpoint,
                                                           "Encoder_Layer_{}_Time_{}_checkpoint.pth".format(layer_id,
                                                                                                            get_time())))
        torch.save(self.decoder.state_dict(), os.path.join(self.decoder_checkpoint,
                                                           "Decoder_Layer_{}_Time_{}_checkpoint.pth".format(layer_id,
                                                                                                            get_time())))
        torch.save(self.D.state_dict(), os.path.join(self.D_checkpoint,
                                                       "D_Layer_{}_Time_{}_checkpoint.pth".format(layer_id, get_time())))

    def train_layer(self, layer_id):
        global loss
        DISP_FREQ = self.DISP_FREQs[layer_id]
        # 分层训练逐步生成：64->128->256->512
        min_average_ssim = -(math.inf)
        for epoch in tqdm(range(self.each_step_epoch[layer_id])):
            self.encoder.train()
            self.decoder.train()
            # print('GAN Epoch [{}/{}]'.format(epoch, self.each_step_epoch[layer_id]))
            for idx, batch in enumerate(self.train_dataloader):
                finding = batch['finding'].to(self.device)
                impression = batch['impression'].to(self.device)
                image = batch['image'].to(self.device)
                loss, generate_image, real_image = self.Loss_on_layer(image, finding, impression, layer_id,
                                                                      self.decoder)

                self.writer.add_scalar('Train_front {}_loss'.format(layer_id),
                                       loss.item(),
                                       epoch * len(self.train_dataloader) + idx)
                # write to tensorboard
                self.writer.add_images("Train_front_{}_Original".format(layer_id),
                                       deNorm(real_image),
                                       epoch * len(self.train_dataloader) + idx)
                self.writer.add_images("Train_front_{}_Predicted".format(layer_id),
                                       deNorm(generate_image),
                                       epoch * len(self.train_dataloader) + idx)
            self.G_lr_scheduler.step(epoch)  # 学习率更新
            # print('loss{:.3f}'.format(loss.item()))
            # self.save_model(layer_id=layer_id)
            if generate_image.shape[2] == args.image_size[0]:  # 只有当图片大小是我们预期大小的时候才开始执行保存操作
                average_ssim = self.predict(layer_id, image)
                print('Epoch:{}--------average_ssim:{:.3f} loss:{:.3f}'.format(epoch, average_ssim, loss.item()))
                if average_ssim > min_average_ssim:
                    min_average_ssim = average_ssim
                    print('save_best_model! Epoch:', epoch)
                    self.save_model(layer_id=layer_id)

    def Loss_on_layer_GAN(self, image, finding, impression, layer_id, decoder, D):
        image = F.interpolate(image, size=(2 ** layer_id) * self.base_size)
        txt_emded, hidden = self.encoder(finding, impression)
        pre_image = decoder(txt_emded, layer_id)
        # train Discriminator
        for _ in range(self.D_step):
            self.D_optimizer.zero_grad()
            pre_fake = D(pre_image, txt_emded)  # [96, 1, 3, 3]
            pre_real = D(image, txt_emded)  # [96, 1, 3, 3]

            gradient_penalty, gradients = cal_gradient_penalty(D, image, pre_image, txt_emded, "cuda",
                                                               lambda_gp=self.lambda_gp[layer_id])

            D_loss = pre_fake.mean() - pre_real.mean() + gradient_penalty

            D_loss.backward(retain_graph=True)

            self.D_optimizer.step()
        # Train Generator
        for _ in range(self.G_step):
            self.G_optimizer.zero_grad()

            pre_fake = D(pre_image, txt_emded)
            adv_loss = - self.adv_loss_ratio * pre_fake.mean()  # 1*判别器假平均值
            adv_loss.backward(retain_graph=True)

            # self.pix_loss_radio=100
            content_loss = self.pix_loss_ratio * self.G_criterion(pre_image.float(),
                                                                  image.float())
            content_loss.backward(retain_graph=True)
            G_loss = content_loss + adv_loss
            self.G_optimizer.step()
        return D_loss, G_loss, pre_image, image

    def train_GAN_layer(self, layer_id):
        DISP_FREQ = self.DISP_FREQs[layer_id]
        self.encoder.train()
        self.decoder.train()
        for epoch in tqdm(range(self.each_step_epoch[layer_id])):
            for idx, batch in enumerate(self.train_dataloader):
                finding = batch['finding'].to(self.device)
                impression = batch['impression'].to(self.device)
                image = batch['image'].to(self.device)
                D_loss, G_loss, generate_image, real_image = self.Loss_on_layer_GAN(image, finding, impression,
                                                                                    layer_id,
                                                                                    self.decoder, self.D)
                self.G_optimizer.zero_grad()

                self.G_optimizer.step()

                # print(epoch * len(self.train_dataloader) + idx)
                self.writer.add_scalar('GAN_G_train_Layer {}_loss'.format(layer_id),
                                       G_loss.item(),
                                       epoch * len(self.train_dataloader) + idx)
                # write to tensorboard
                self.writer.add_scalar("GAN_D_train_Layer{}_Original".format(layer_id),
                                       D_loss.item(),
                                       epoch * len(self.train_dataloader) + idx)
                self.writer.add_images("Train_front_{}_Predicted".format(layer_id),
                                       deNorm(generate_image),
                                       epoch * len(self.train_dataloader) + idx)
                self.writer.add_images("GAN_Train_Original_front_{}".format(layer_id),
                                       deNorm(real_image),
                                       epoch * len(self.train_dataloader) + idx)
            self.G_lr_scheduler.step(epoch)
            self.D_lr_scheduler.step(epoch)
            if generate_image.shape[2] == args.image_size[0]:  # 只有当图片大小是我们预期大小的时候才开始执行保存操作
                average_ssim = self.predict(layer_id, image)
                print('Epoch:{}--------average_ssim:{:.3f} loss:{:.3f}'.format(epoch, average_ssim, loss.item()))
                if average_ssim > min_average_ssim:
                    min_average_ssim = average_ssim
                    print('save_best_model! Epoch:', epoch)
                    self.save_model2(layer_id=layer_id)
            else:
                if (epoch + 1) % 20 == 0 and epoch != 0:
                    self.save_model2(layer_id=layer_id)

    def predict(self, layer_id, image):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            ssim_score_list = []
            for idx, batch in enumerate(self.test_dataloader):
                finding = batch['finding'].to(self.device)
                impression = batch['impression'].to(self.device)
                txt_emded, hidden = self.encoder(finding, impression)
                generate_image = self.decoder(txt_emded, layer_id)
                real_image = F.interpolate(image, size=(2 ** layer_id) * self.base_size)
                if real_image.shape[0] != generate_image.shape[0]:
                    repeat_size = (generate_image.shape[0] // real_image.shape[0]) + 1
                    real_image = real_image.repeat(repeat_size, 1, 1, 1)[:generate_image.shape[0], :]
                ssim_score = ssim(generate_image, real_image)
                ssim_score_list.append(ssim_score)
            average_ssim = sum(ssim_score_list) / len(ssim_score_list)
            return average_ssim

    def train(self):
        for layer_id in range(self.P_ratio + 1):
            self.define_Discriminator(layer_id)
            self.define_dataloader(layer_id)
            self.define_opt(layer_id)

            print("Start training GAN {}".format(layer_id))
            # self.train_layer(layer_id)
            self.train_GAN_layer(layer_id)


def main():
    trainer = BaseTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
