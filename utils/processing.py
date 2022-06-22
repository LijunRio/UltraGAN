import pydicom as dcm
import matplotlib.pyplot as plt
import unicodedata
import re
import numpy as np
import datetime
import xml.etree.ElementTree as ET
import cv2
import torch
from torchvision import transforms
import os
import random
import json


# Not_list = ['not','no','none','non']
# syno_f = open('./utils/syno_dict.json')
# syno_dict = json.load(syno_f)
#
# anto_f = open('./utils/anto_dict.json')
# anto_dict = json.load(anto_f)

def read_png(file_name):
    '''Read Png image'''
    img = cv2.imread(file_name)
    # img = cv2.resize(img,(512,512))
    # new_fn = '/_'.join(os.path.split(file_name))
    # cv2.imwrite(new_fn,img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def read_dcm(file_name):
    dcm_data = dcm.dcmread(file_name)
    return dcm_data.pixel_array


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    '''
    Clean the text data
    Lower letter, remove numbers and remove space
    '''
    s = unicodeToAscii(s.lower().strip())
    s = re.sub("[^a-zA-Z0-9_]", r" ", s)
    s = s.replace(' ', '')
    return s.strip()


def normalizeSentence(s):
    '''
    Clean the text data
    Lower letter, remove numbers and remove space
    '''
    s = unicodeToAscii(s.lower().strip())
    s = re.sub("[^a-zA-Z0-9_]", r" ", s)
    return s.strip()


def get_time():
    '''Get local time for checkpoint saving'''
    return (str(datetime.datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def deNorm(output):
    # [-1,1] -> [0,1]
    return (output + 1) / 2


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    '''Visualize tensor in the tensorboard'''
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def read_XML(xmlfile):
    tree = ET.parse(xmlfile)
    raw_report = tree.find("./MedlineCitation/Article/Abstract")
    report_dict = {}
    for child in raw_report:
        report_dict[child.attrib['Label']] = child.text
    txt = report_dict["FINDINGS"].split() + report_dict["IMPRESSION"].split()
    report = [normalizeString(s) for s in txt]
    return report


def read_XML_random_anto(xmlfile, p=0):
    tree = ET.parse(xmlfile)
    raw_report = tree.find("./MedlineCitation/Article/Abstract")
    report_dict = {}
    for child in raw_report:
        report_dict[child.attrib['Label']] = child.text
    finding = report_dict["FINDINGS"].split()
    impre = report_dict["IMPRESSION"].split()
    nfinding = []
    nimpre = []
    for s in finding:

        if s in anto_dict.keys():
            if random.random() < p:
                nfinding.append(random.sample(anto_dict[s], 1)[0])
                continue
        nfinding.append(normalizeString(s))

    for s in impre:
        if s in anto_dict.keys():
            if random.random() < p:
                nimpre.append(random.sample(anto_dict[s], 1)[0])
                continue
        nimpre.append(normalizeString(s))

    return nfinding, nimpre


def read_XML2(xmlfile):
    tree = ET.parse(xmlfile)
    raw_report = tree.find("./MedlineCitation/Article/Abstract")
    report_dict = {}
    for child in raw_report:
        report_dict[child.attrib['Label']] = child.text
    finding = report_dict["FINDINGS"].split()
    impre = report_dict["IMPRESSION"].split()
    finding = [normalizeString(s) for s in finding]
    impre = [normalizeString(s) for s in impre]
    return finding, impre


def read_XML_sentence(xmlfile):
    tree = ET.parse(xmlfile)
    raw_report = tree.find("./MedlineCitation/Article/Abstract")
    report_dict = {}
    for child in raw_report:
        report_dict[child.attrib['Label']] = child.text
    finding_sent = report_dict["FINDINGS"].split('.')
    impre_sent = report_dict["IMPRESSION"].split('.')
    finding = []
    impre = []
    for s in finding_sent:
        find_s = []
        for w in s.split():
            # print(w)
            find_s.append(normalizeString(w))
        finding.append(find_s)
    for s in impre_sent:
        imp_s = []
        for w in s.split():
            imp_s.append(normalizeString(w))
        impre.append(imp_s)
    return finding, impre


def test_XML(xmlfile):
    tree = ET.parse(xmlfile)
    raw_report = tree.find("./MedlineCitation/Article/Abstract")
    for child in raw_report:
        if child.text == None:
            return False
    return True


def find_parentImage(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    image_names = []
    for image in root.findall("./parentImage"):
        image_names.append(image.attrib['id'])
    return image_names


class Rescale(object):
    """Rescale the image in the sample to a given size

    Args:
        Output_size(tuple): Desired output size
            tuple for output_size
    """

    def __init__(self, output_sizes):
        # assert isinstance(output_sizes, tuple)
        new_h, new_w = output_sizes
        self.resize = (int(new_h), int(new_w))

    def __call__(self, image):
        img = cv2.resize(image, dsize=self.resize, interpolation=cv2.INTER_CUBIC)

        return img


class ToTensor(object):
    """Convert darray in sample to Tensors"""

    def __call__(self, image):
        # torch image: channel * H * W
        h, w = image.shape[:2]
        image = image.reshape((1, h, w)) / 255
        image = (image - 0.5) / 0.5
        return image


class Equalize(object):
    def __init__(self, mode="Normal"):
        self.mode = mode
        self.equlizer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, image):
        if self.mode == "Normal":
            equ = cv2.equalizeHist(image)
        elif self.mode == "CLAHE":
            equ = self.equlizer.apply(image)
        return equ


class Rescale2(object):
    """Rescale the image in the sample to a given size

    Args:
        Output_size(tuple): Desired output size
            tuple for output_size
    """

    def __init__(self, output_sizes):
        assert isinstance(output_sizes, tuple)
        new_h, new_w = output_sizes
        self.resize = (int(new_h), int(new_w))

    def __call__(self, sample):
        image_F = sample['image_F']

        img_F = cv2.resize(image_F, dsize=self.resize, interpolation=cv2.INTER_CUBIC)

        image_L = sample['image_L']

        img_L = cv2.resize(image_L, dsize=self.resize, interpolation=cv2.INTER_CUBIC)

        return {
            'subject_id': sample['subject_id'],
            'finding': sample['finding'],
            'impression': sample['impression'],
            'image_F': img_F,
            'image_L': img_L,
            'len': sample['len']}


class ToTensor2(object):
    """Convert darray in sample to Tensors"""

    def __call__(self, sample):
        image_F, image_L = sample['image_F'], sample['image_L']

        # torch image: channel * H * W
        h, w = image_F.shape[:2]
        image_F = image_F.reshape((1, h, w)) / 255
        image_F = (image_F - 0.5) / 0.5

        image_L = image_L.reshape((1, h, w)) / 255
        image_L = (image_L - 0.5) / 0.5
        return {'subject_id': torch.tensor(sample['subject_id'], dtype=torch.long),
                'finding': torch.tensor(sample['finding'], dtype=torch.long),
                'impression': torch.tensor(sample['impression'], dtype=torch.long),
                'image_F': torch.tensor(image_F, dtype=torch.float),
                'image_L': torch.tensor(image_L, dtype=torch.float),
                'len': torch.tensor(sample['len'], dtype=torch.long)}


class Equalize2(object):

    def __call__(self, sample):
        image_F, image_L = sample['image_F'], sample['image_L']
        image_F = cv2.equalizeHist(image_F)
        image_L = cv2.equalizeHist(image_L)

        return {
            'subject_id': sample['subject_id'],
            'finding': sample['finding'],
            'impression': sample['impression'],
            'image_F': image_F,
            'image_L': image_L,
            'len': sample['len']}
