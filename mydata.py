# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 16:28:42 2018

@author: Baek
"""

import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from os import listdir
from os.path import join
from PIL import Image
import random
import matplotlib.pyplot as plt
import imageio
import cv2
import numpy as np
import torch
from skimage import color
from data2 import common

def is_img_file(filename):
    return any(filename.endswith(extension) for extension in [".hdr"])

def load_img(filepath):
    img = Image.open(filepath)
    #img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, args, name='DIV2K' ,train=True):
        super(DatasetFromFolder, self).__init__()
        self.args = args
        root_dir = join(args.dir_data, name)
        if name == 'DIV2K':
            LR_dir = join(root_dir, 'DIV2K_train_LR_bicubic')
            LR_dir = join(LR_dir, 'X'+str(args.scale))

            HR_dir = join(root_dir, 'DIV2K_train_HR')
        r = args.data_range.split('/')
        if train:
            data_range = r[0].split('-')
        elif args.test_only:
            data_range = r[0].split('-')
        else:    
            data_range = r[1].split('-')

        HR_names = sorted(listdir(HR_dir))
        HR_names = HR_names[int(data_range[0])-1:int(data_range[1])]
        LR_names = sorted(listdir(LR_dir))
        LR_names = LR_names[int(data_range[0])-1:int(data_range[1])]
        self.LR_filenames = [join(LR_dir, x) for x in LR_names]
        self.HR_filenames = [join(HR_dir, x) for x in HR_names]
            
        self.patch_size = args.patch_size
        self.train = train
        
    def __getitem__(self, index):
        LR = load_img(self.LR_filenames[index])
        HR = load_img(self.HR_filenames[index])

        LR, HR = np.asarray(LR), np.asarray(HR)
        if self.train:
            LR, HR = random_crop(LR, HR, patch_size = self.args.patch_size)
            #LR, HR = self.get_patch(LR, HR)
            LR, HR = augment(LR, HR)

        LR = torch.from_numpy((LR.transpose([2, 0, 1])).copy())
        HR = torch.from_numpy((HR.transpose([2, 0, 1])).copy())
        LR = LR.type(torch.FloatTensor)
        HR = HR.type(torch.FloatTensor)
        filename = self.HR_filenames[index]

        return LR, HR, filename

    def __len__(self):
        return len(self.LR_filenames)

    def get_patch(self, lr, hr):
        scale = self.args.scale
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(scale > 1),
                input_large=False
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

def get_training_set(args):

    return DatasetFromFolder(args, 'DIV2K', train=True)

def get_val_set(args):

    return DatasetFromFolder(args, 'DIV2K', train=False)


def random_crop(LR, HR, patch_size = 96):
    h, w, c = np.shape(HR)

    crop_w = patch_size
    crop_h = patch_size
    i = random.randint(0, h- crop_h)
    j = random.randint(0, w - crop_w)
    LR = LR[i//2:(i+crop_h)//2, j//2:(j+crop_w)//2,:]
    HR = HR[i:i+crop_h, j:j+crop_w, :]
    
    return LR, HR

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):

        if hflip: 
            img = img[:, ::-1, :]
        if vflip: 
            img = img[::-1, :, :]
        #if rot90: img = img.transpose(1,0,2)
        if rot90: 
            img = np.rot90(img, 1, (0,1))
        return img

    return [_augment(a) for a in args]