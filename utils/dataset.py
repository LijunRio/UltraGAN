import os
import json

import jieba
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


# class BaseDataset(Dataset):
#     def __init__(self, args, split, transform=None):
#         self.ann_path = args.ann_path
#         self.dict_pth = args.vocab_path
#         self.split = split
#         word_dict = json.loads(open(self.dict_pth, 'r', encoding="utf_8_sig").read())
#         self.vocab = word_dict[0]
#         self.max_finding_len = word_dict[3]
#         self.max_impression_len = word_dict[2]
#         self.transform = transform
#         self.ann = json.loads(open(self.ann_path, 'r', encoding="utf_8_sig").read())  # load json格式的注释
#
#         self.examples = self.ann[self.split]  # split-> train valid test
#         for i in range(len(self.examples)):
#             # 新建一个ids为token后端内容
#             finding_list = list(jieba.cut(self.examples[i]['finding']))
#             txt_finding_sen = [self.vocab[w] for w in finding_list]
#             txt_finding = np.pad(txt_finding_sen, (self.max_finding_len - len(txt_finding_sen), 0),
#                                  'constant', constant_values=0)
#
#             impression_list = list(jieba.cut(self.examples[i]['impression']))
#             txt_impression_sen = [self.vocab[w] for w in impression_list]
#             txt_impression = np.pad(txt_impression_sen, (self.max_impression_len - len(txt_impression_sen), 0),
#                                     'constant', constant_values=0)
#             self.examples[i]['fi_ids'] = txt_finding
#             self.examples[i]['im_ids'] = txt_impression
#
#     def __len__(self):
#         return len(self.examples)
#
#
# class MyDataset(BaseDataset):
#     def __getitem__(self, idx):
#         example = self.examples[idx]
#         image_path = example['image_path']
#         image = Image.open(image_path).convert('RGB')
#         if self.transform is not None:
#             image = self.transform(image)
#         finding_ids = example['fi_ids']  # report的token
#         impression_ids = example['im_ids']
#         image_id = example['uid']
#         # sample = {
#         #     'finding': torch.tensor(finding_ids, dtype=torch.long),
#         #     'impression': torch.tensor(impression_ids, dtype=torch.long),
#         #     'image': torch.tensor(image, dtype=torch.float),
#         # }
#         sample = (image_id, image, finding_ids, impression_ids)
#         return sample


class MyDataset2(Dataset):
    def __init__(self, args, split, transform=None):

        self.ann_path = args.ann_path
        self.dict_pth = args.vocab_path
        self.split = split
        self.transform = transform
        word_dict = json.loads(open(self.dict_pth, 'r', encoding="utf_8_sig").read())
        self.vocab = word_dict[0]
        self.max_finding_len = word_dict[3]
        self.max_impression_len = word_dict[2]
        self.max_word_fi = word_dict[5]
        self.max_word_im = word_dict[4]
        self.image_path = args.image_path
        # print(self.max_word_im, self.max_word_fi, self.max_impression_len, self.max_finding_len)
        self.ann = json.loads(open(self.ann_path, 'r', encoding="utf_8_sig").read())  # load json格式
        self.examples = self.ann[self.split]  # split-> train valid test

        # new_define
        self.findings = []
        self.impression = []
        self.image_pths = []
        self.subject_ids = []

        for i in range(len(self.examples)):
            image_pth = os.path.join(self.image_path, self.examples[i]['image_path'])
            fi = self.examples[i]['finding']
            im = self.examples[i]['impression']
            fi = fi.split('。')
            im = im.split('\n')
            txt_finding = []
            txt_impression = []
            for sen in fi:
                sen_list = list(jieba.cut(sen))
                txt_finding_sen = [self.vocab[w] for w in sen_list]
                txt_finding_sen = np.pad(txt_finding_sen, (self.max_word_fi - len(txt_finding_sen), 0),
                                         'constant', constant_values=0)
                txt_finding.append(txt_finding_sen)
            for sen in im:
                sen_list = list(jieba.cut(sen))
                txt_impression_sen = [self.vocab[w] for w in sen_list]
                txt_impression_sen = np.pad(txt_impression_sen,
                                            (self.max_word_im - len(txt_impression_sen), 0), 'constant',
                                            constant_values=0)
                txt_impression.append(txt_impression_sen)

            txt_impression = np.pad(np.array(txt_impression),
                                    ((self.max_impression_len - len(txt_impression), 0), (0, 0)), 'constant',
                                    constant_values=0)

            txt_finding = np.pad(np.array(txt_finding), ((self.max_finding_len - len(txt_finding), 0), (0, 0)),
                                 'constant', constant_values=0)

            self.findings.append(txt_finding)
            self.impression.append(txt_impression)
            self.image_pths.append(image_pth)
            self.subject_ids.append(self.examples[i]['uid'])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = np.array(Image.open(self.image_pths[idx]).convert('L'))

        if self.transform:
            image = self.transform(image)

        sample = {
            'subject_id': torch.tensor(self.subject_ids[idx], dtype=torch.long),
            'finding': torch.tensor(self.findings[idx], dtype=torch.long),
            'impression': torch.tensor(self.impression[idx], dtype=torch.long),
            'image': torch.tensor(image, dtype=torch.float),
        }
        return sample
