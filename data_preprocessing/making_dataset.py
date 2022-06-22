import os

import matplotlib.pyplot as plt
import pandas
from tqdm import tqdm
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
import pandas as pd
import glob
import cv2
from PIL import Image
import copy
import json


def read_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    report_dict = {}
    for child in root:
        key = child.find("./key/string").text
        value = child.find("./value/string").text
        if key not in report_dict:
            report_dict.update({key: [value]})
    return report_dict


def MergeTwoDict(d1, d2):
    # d1 = {'a':[0,1], 'b':[1,2], 'd':[1,3]}
    # d2 = {'a':[0], 'c':[1]}
    key_set = set(list(d1.keys()) + list(d2.keys()))
    value = set(d1.keys() ^ d2.keys())  # find the differences between two dicts
    first_key = list(d1.keys())[0]
    len_key = len(d1[first_key])
    for key in value:
        if key not in d1:
            d1.update({key: ['na'] * len_key})

    for key in d1:
        if key in d2:
            d1[key].append(d2[key][0])
        else:
            d1[key].append('NA')
    return d1


def read_all_xml(root_path):
    folder_list = os.listdir(root_path)
    total_dict = {}

    for item in tqdm(folder_list):
        item_list = item.split('(')
        uid = item_list[0]
        name = item_list[1][:-1]
        xml_file = os.path.join(root_path, item + '\\info.xml')
        report_dict = {'uid': [uid], 'name': [name]}
        report_dict.update(read_xml(xml_file))
        if len(total_dict) == 0:
            total_dict = report_dict
        else:
            total_dict = MergeTwoDict(total_dict, report_dict)
    return total_dict


"""
step 1, read xml and save to csv
we have directly delete some columns by excel, and save the file as my_report.csv.
"""
root_path = 'D:\\RIO\\All_Datastes\\甲状腺超声数据集'
total_dict = read_all_xml(root_path)
df = pd.DataFrame.from_dict(total_dict)
df.to_csv('report_new.csv', encoding="utf_8_sig")

"""
step 2, First selet the data, whose device value is '通用VOLUSON730'. Total numbers:672
save the images pth to img_pth1 [img1, img2, ..., img]
Save the result to 'deviceVOLUSON730.csv'
"""
# root_path = 'D:/RIO/All_Datastes/甲状腺超声数据集/'
# df = pd.read_csv('my_report.csv', encoding='gb18030')
# df_new = df[df['检查设备'] == '通用VOLUSON730']
# keys = df_new.columns
# keys = list(keys) + ['img_pth1', 'img_pth2']
# final_df = pd.DataFrame(columns=keys)
# for index, item in tqdm(df_new.iterrows()):
#     uid = item['uid']
#     name = item['name']
#     folder_pth = os.path.join(root_path, '00' + str(uid) + '(' + name + ')')
#     img_files = glob.glob(folder_pth + '/*.jpg')
#     item['img_pth1'] = img_files
#     final_df = final_df.append(item, ignore_index=True)
# final_df.to_csv('deviceVOLUSON730_2.csv', encoding="utf_8_sig")

"""
step 3, crop the images
use opencv to crop the image and fill the blank with black pixels.
chose the first process images-reports as final data pair
"""


def check_empty_img(pth):
    image = Image.open(pth)
    flag = False
    if image is not None:
        flag = True
    return flag


def resize_donot_change_radio(region, x1, x2, y1, y2):
    # select the longest edge resized to 512
    w, h = x2 - x1, y2 - y1  # h, w = image.shape
    m = max(w, h)
    ratio = 512.0 / m
    new_w, new_h = int(ratio * w), int(ratio * h)
    assert new_w > 0 and new_h > 0
    resized = cv2.resize(region, (new_w, new_h))

    # padded the resized regoin to 512 and 512
    W, H = 512, 512
    top = (H - new_h) // 2
    bottom = (H - new_h) // 2
    if top + bottom + h < H:
        bottom += 1

    left = (W - new_w) // 2
    right = (W - new_w) // 2
    if left + right + w < W:
        right += 1
    pad_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return pad_image


def detect_red_blue_green(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_red = np.array([-20, 100, 100])
    upper_red = np.array([13, 255, 255])

    # Here we are defining range of bluecolor in HSV
    # This creates a mask of blue coloured
    # objects found in the frame.
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = mask1 | mask2
    contours2, hierarchy2 = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours2)


def find_crop_box(pth):
    flag = False
    final_region = None
    if check_empty_img(pth):
        im = Image.open(pth)
        opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        # bgr->gray->threshold->erode->dilate
        gray = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 15, 255, 0)
        kernel = np.ones((7, 7), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=3)
        thresh = cv2.dilate(thresh, kernel, iterations=3)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_rect = None
        for index in range(len(contours)):
            contour = contours[index]
            area = cv2.contourArea(contour)
            rect = cv2.minAreaRect(contour)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
            if area > max_area:
                max_area = area
                max_rect = rect
        box = cv2.boxPoints(max_rect)
        box = np.int0(box)
        sort_box = box[box[:, 1].argsort()]
        # get up two vertex and bottom vertex -> crop images and get the roi region
        upVertex = sort_box[:2]
        bottomVertex = sort_box[2:]
        upVertex_s = upVertex[upVertex[:, 0].argsort()]
        bottomVertex_s = bottomVertex[bottomVertex[:, 0].argsort()]
        y1, y2 = upVertex_s[0][1], bottomVertex_s[0][1]
        x1, x2 = upVertex_s[0][0], upVertex_s[1][0]
        # if prevent bot function give a negative value
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        region = opencvImage[y1:y2, x1:x2]
        # pad_image = resize_donot_change_radio(region, x1, x2, y1, y2)
        resize_region = cv2.resize(region, (512, 512))
        len_contours = detect_red_blue_green(resize_region)
        if len_contours < 2:
            flag = True
            final_region = resize_region
    else:
        print('PTH is empty!')
    return flag, final_region


# from ast import literal_eval
#
# new_root = 'D:/RIO/All_Datastes/甲状腺处理后图片/'
# final_df = pd.read_csv('deviceVOLUSON730_2.csv', encoding="utf_8_sig")
# keys = final_df.columns
# keys = list(keys) + ['image_pth']
# keys = [x for x in keys if x not in ['img_pth1', 'img_pth2']]
# new_df = pd.DataFrame(columns=keys)
# for index, item in tqdm(final_df.iterrows()):
#     pth_list = literal_eval(item['img_pth1'])
#     new_img = None
#     for pth in pth_list:
#         flag, region = find_crop_box(pth)
#         if flag:
#             new_img = region
#         else:
#             continue
#     if new_img is not None:
#         img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
#         im_pil = Image.fromarray(img)
#         new_pth = os.path.join(new_root, item['name'])+ '.jpg'
#         item['image_pth'] = new_pth
#         new_df = new_df.append(item, ignore_index=True)
#         im_pil.save(new_pth)
# new_df.to_csv('final_report_voluson.csv', encoding="utf_8_sig")

"""
step 4, build the vocabulary dictionary
use jieba to split chinese characters
Finally, the vocabulary dictionary is saved as 'dict.json'
"""
# import jieba
# file_name = 'D:/ultrasonic_project/ulstrasonic_code/data_preprocessing/final_report_voluson.csv'
# data = pd.read_csv(file_name, encoding="utf_8_sig")
# vocab = {}
# sen_len_finding = []
# sen_len_impression = []
# each_fi_len = []  # store each sample' sentence length
# each_im_len = []
# for index, item in tqdm(data.iterrows()):
#     finding = item['检查所见']
#     impression = item['诊断印象']
#     finding = finding.split('。')
#     impression = impression.split('\n')
#
#     for sen in finding:
#         sen_list = list(jieba.cut(sen))
#         each_fi_len.append(len(sen_list))
#         for item in sen_list:
#             if item not in vocab:
#                 vocab.update({item: 1})
#             else:
#                 vocab[item] += 1
#     for sen in impression:
#         sen_list = list(jieba.cut(sen))
#         each_im_len.append(len(sen_list))
#         for item in sen_list:
#             if item not in vocab:
#                 vocab.update({item: 1})
#             else:
#                 vocab[item] += 1
#
#     # record length information
#     sen_len_finding.append(len(finding))
#     sen_len_impression.append(len(impression))
#
# word_to_idx = {}
# count = 1
# for i, word in enumerate(vocab):
#     if word in word_to_idx.keys():
#         pass
#     else:
#         word_to_idx[word] = count
#         count += 1
#
# vocab_len = count + 1
# max_len_im, max_len_fi = max(sen_len_impression), max(sen_len_finding)
# max_word_im, max_word_fi = max(each_im_len), max(each_fi_len)
# print("Totally {} vocabulary".format(vocab_len))
# print("Max Finding length {}".format(max_len_fi))
# print("Max Impression length {}".format(max_len_im))
# word_dict = 'dict.json'
# with open(word_dict, 'w', encoding="utf_8_sig") as f:
#     json.dump([word_to_idx, vocab_len, max_len_im, max_len_fi, max_word_im, max_word_fi], f, ensure_ascii=False)
"""
step 5, save to new json file
"""
# data = pd.read_csv( 'D:/ultrasonic_project/ulstrasonic_code/data_preprocessing/final_report_voluson.csv',
#                     encoding="utf_8_sig")
# train = []
# test = []
# val = []
# train_data = data[:463]
# test_data = data[463:596]
# val_data = data[596:]
# dataset = [train_data, test_data, val_data]
# dataset_list = [train, test, val]
# split_list = ['train', 'test', 'val']
# for i in tqdm(range(len(dataset))):
#     cur_set = dataset[i]
#     tmp = []
#     for index, item in cur_set.iterrows():
#         uid = item['uid']
#         finding = item['检查所见']
#         impression = item['诊断印象']
#         name = item['name']
#         split = split_list[i]
#         tmp.append({'name':name, 'uid':uid, 'finding':finding, 'impression':impression, 'image_path':item['image_pth'], 'split':split})
#     dataset_list[i] = tmp
# final_save = {}
# for i in range(len(dataset_list)):
#     final_save.update({split_list[i]:dataset_list[i]})
# with open('annotation.json', 'w', encoding="utf_8_sig") as f:
#     json.dump(final_save, f, ensure_ascii=False)


"""
step6: select and resize
"""
# pth = 'D:\\RIO\\All_Datastes\\超声筛选替换图像'
# file = os.listdir(pth)
# for item in file:
#     img_pth = os.path.join(pth, item)
#     im = Image.open(img_pth)
#     opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
#     resize_img = cv2.resize(opencvImage, (512, 512))
#
#     img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
#     im_pil = Image.fromarray(img)
#     im_pil.save(img_pth)

"""
rename data
"""
# pth = "D:/RIO/All_Datastes/甲状腺处理后图片"
# new_pth = "D:/RIO/All_Datastes/ulstroasonic_images/"
# ann_path = 'D:/ultrasonic_project/ulstrasonic_code/data_preprocessing/annotation.json'
# ann = json.loads(open(ann_path, 'r', encoding="utf_8_sig").read())
# split_list = ['train', 'test', 'val']
# for split in split_list:
#     for dataset in ann[split]:
#        image_path = dataset['image_path']
#        im = Image.open(image_path)
#        new_name = str(dataset['uid']) + '.jpg'
#        pth_new = os.path.join(new_pth, new_name)
#        dataset['image_path'] = new_name
#        im.save(pth_new)
# with open('annotation2.json', 'w', encoding="utf_8_sig") as f:
#     json.dump(ann, f, ensure_ascii=False)
