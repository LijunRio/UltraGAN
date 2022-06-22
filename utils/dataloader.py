import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .dataset import MyDataset


class MyDataloader(DataLoader):
    def __init__(self, args, split, shuffle):
        self.args = args
        self.dataset_name = MyDataset
        self.batch_size = args.batch_size
        self.shuffle = shuffle  # shuffle = True时， 每个epoch都会打乱数据集
        self.num_workers = args.num_workers
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        self.dataset = MyDataset(self.args, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    # 使用@staticmethod或@classmethod，就可以不需要实例化，直接类名.方法名()来调用。
    """
    其实，collate_fn可理解为函数句柄、指针...或者其他可调用类(实现__call__函数)。 函数输入为list，list中的元素为欲取出的一系列样本。
    总的来说，就是实现如何去样本的功能
    """

    @staticmethod
    def collate_fn(data):  # 用于样本不定长
        """
         data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
        """
        images_id, image, finding_ids, impression_ids = zip(*data)
        finding = torch.tensor(finding_ids, dtype=torch.long)
        impression = torch.tensor(impression_ids, dtype=torch.long)

        return images_id, image, finding, impression

