__author__ = 'Rio'

from yacs.config import CfgNode as CN

config = CN()
config.ann_path = 'D:/ultrasonic_project/ulstrasonic_code/data_preprocessing/annotation2.json'
config.vocab_path = 'D:/ultrasonic_project/ulstrasonic_code/data_preprocessing/dict.json'
config.image_path = 'D:/RIO/All_Datastes/ulstroasonic_images/'
# config.batch_size = [96, 24, 6, 1]#[96, 32, 12, 6]
config.batch_size = [96,48,24,12]

config.G_LR = [0.0003, 0.0003, 0.0002, 0.0001]
config.num_workers = 0  # In windows this parameters set to 0
config.beta1 = 0.9
config.lr_decay_epoch = [[45], [45, 70], [45, 70, 90], [45, 70, 90]]
# config.D_LR = [0.0003, 0.0003, 0.0002, 0.0001]  # Discriminator learning rate
config.D_LR = [0.0003, 0.0003, 0.0002, 0.0001]  # Discriminator learning rate
# config.each_step_epoch = [0, 90, 90, 120]
config.each_step_epoch = [200,400,600,120]
config.content_loss = "L1"
config.adv_loss_ratio = 1
config.pix_loss_ratio = 100
config.d_step = 1
config.g_step = 1

config.embed_size = 128
config.hidden_size = 128
config.rnn_cell = 'LSTM'
config.d_channel_size = 512
config.base_size = 64  # 32  # do not change
config.image_size = [512, 512]  # setting the output image size
config.n_gpu = 1
config.exp_name = "Save_Outputs"

config.resume_encoder = './checkpoint/mygan/encoder/Encoder_Layer_0_Time_2022-03-16-22-15_checkpoint.pth'
config.resume_decoder = './checkpoint/MYGAN/decoder/Decoder_Layer_0_Time_2022-03-16-22-15_checkpoint.pth'
# config.resume_encoder = ''
# config.resume_decoder = ''

config.encoder_checkpoint = "./checkpoint/Generator/encoder"
config.decoder_checkpoint = "./checkpoint/Generator/decoder"
config.D_checkpoint = "./checkpoint/OPENI/XRayGAN2/D"


