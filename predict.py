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
from utils.processing import Rescale, ToTensor, Equalize, deNorm
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from tqdm import tqdm
from config import config as args
from torchvision.utils import save_image
import os


class Tester(object):
    def __init__(self, args):
        self.args = args
        self.device, self.device_ids = self._prepare_device(args.n_gpu)
        self.exp_name = args.exp_name
        self.dict_pth = args.vocab_path
        word_dict = json.loads(open(self.dict_pth, 'r', encoding="utf_8_sig").read())
        self.vocab = word_dict[0]
        self.max_finding_len = word_dict[3]
        self.max_impression_len = word_dict[2]
        self.batch_size = args.batch_size
        self.beta1 = self.args.beta1
        # self.max_epoch = self.args.max_epoch
        self.image_size = args.image_size
        self.base_size = args.base_size
        self.encoder_resume = args.resume_encoder
        self.decoder_resume = args.resume_decoder

        self.testset = MyDataset2(args, split='test', transform=transforms.Compose([
            Rescale(self.image_size),
            Equalize(),
            ToTensor()]))

        self.test_dataloader = DataLoader(self.testset,
                                          batch_size=12,
                                          shuffle=False,
                                          drop_last=True)
        self.save_img_dir = './save_image/test1'
        if os.path.exists(self.save_img_dir) == False:
            os.mkdir(self.save_img_dir)

        self.base_ratio = int(np.log2(self.base_size))
        self.P_ratio = int(np.log2(self.image_size[0] // self.base_size))
        self.define_nets()
        self.encoder = nn.DataParallel(self.encoder).cuda()
        self.decoder = nn.DataParallel(self.decoder).cuda()

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

    def define_opt(self, layer_id):
        '''Define optimizer'''
        self.G_optimizer = torch.optim.Adam([{'params': self.encoder.parameters()}] +
                                            [{'params': self.decoder.parameters()}],
                                            lr=self.args.G_LR[layer_id], betas=(self.beta1, 0.999))
        self.G_lr_scheduler = MultiStepLR(self.G_optimizer, milestones=self.args.lr_decay_epoch[layer_id], gamma=0.2)

        self.D_optimizer = torch.optim.Adam([{'params': self.D.parameters()}],
                                            lr=self.args.D_LR[layer_id], betas=(self.beta1, 0.999))
        self.D_lr_scheduler = MultiStepLR(self.D_optimizer, milestones=self.args.lr_decay_epoch[layer_id], gamma=0.2)

    def load_model(self):
        self.encoder.load_state_dict(torch.load(self.encoder_resume))
        self.decoder.load_state_dict(torch.load(self.decoder_resume))

    def test(self):
        self.load_model()
        layer_id = self.P_ratio
        self.encoder.eval()
        self.decoder.eval()
        print("Start generating")
        for idx, batch in enumerate(tqdm(self.test_dataloader)):
            finding = batch['finding'].to(self.device)
            impression = batch['impression'].to(self.device)
            txt_emded, hidden = self.encoder(finding, impression)
            pre_image = self.decoder(txt_emded, layer_id)
            pre_image = deNorm(pre_image).data.cpu()
            subject_id = batch['subject_id'].data.cpu().numpy()
            for i in range(pre_image.shape[0]):
                save_image(pre_image[i], '{}/{}.png'.format(self.save_img_dir, subject_id[i]))


def main():
    trainer = Tester(args)
    trainer.test()


if __name__ == '__main__':
    main()
    # pretrained_dict = torch.load(args.resume_encoder)
