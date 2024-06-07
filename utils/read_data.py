import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import logging
import os

from pathlib import Path

from PIL import Image


def read_img(filename:str, resize):
    '''
    @para: 
            filename
            resize -> [w, h]
    @output: img -> [h, w, c]
    '''
    ext = os.path.splitext(filename)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.jpg' or ext == '.tif' or ext == ".tiff":
        img = Image.open(filename)
        w, h = img.size
        if resize!=None :
            if resize[0] != w:
                img = img.resize(resize)
        img = np.array(img)
        if img.dtype == 'uint16' and len(img.shape)==2:
            img = conver_to_uint8(img)
            img = np.expand_dims(img, 2)

    return img

def read_disp_dfc(filename, resize, min_disp, max_disp):
    '''
    @para: resize -> [w, h]
    '''
    disp = Image.open(filename)
    w, h = disp.size
    if resize[0] != w:
        disp = disp.resize(resize)
    disp = np.array(disp)
      
    disp[np.isnan(disp)] = 0

    # img has been resized, thus the disparity should resized
    if resize[0] != w:
        scale = w / resize[0]
        disp = disp / scale

    # generate mask
    valid = np.logical_and((disp != -999), (disp < max_disp))
    valid = np.logical_and(valid, (disp > min_disp))

    # valid = np.ones_like(disp)
    return disp, valid


def read_disp_whu(filename, resize, min_disp, max_disp):
    '''
    @para: resize -> [w, h]
    '''
    disp = Image.open(filename)
    w, h = disp.size
    if resize[0] != w:
        disp = disp.resize(resize)
    disp = np.array(disp)

    # img has been resized, thus the disparity should resized
    if resize[0] != w:
        scale = w / resize[0]
        disp = disp / scale

    # generate mask
    valid = np.logical_and((disp < max_disp), (disp > min_disp))
    # valid = np.ones_like(disp)
    return disp, valid

def img_norm(img:np.array, min_per=0.5) -> torch.tensor :
    """
    @para:
        img -> [h, w, c]
    @output:
        img -> [c, h, w]
    """

    # img = img/255
    # transform = transforms.Compose([transforms.ToTensor(),
    #                             transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])    
    # img = transform(img)
    # img = img.float()
    
    img = img.astype(np.float32)
    for i in range(img.shape[2]):
        img[:,:,i] = (img[:,:,i] - np.mean( img[:,:,i] )) / np.std(img[:,:,i])
    img = torch.from_numpy(img).permute(2,0,1)

    return img

def conver_to_uint8(img):
    min_per = 0.3
    max_pixel = 255
    max_per = 100 - min_per
    cut_min = np.nanpercentile(img.ravel(), min_per) #ravel()方法将数组维度拉成一维数组
    cut_max = np.nanpercentile(img.ravel(), max_per)
    img = np.clip(img, cut_min, cut_max)
    return (max_pixel * (img - cut_min) / (cut_max - cut_min)).astype(np.uint8)  # img.astype(np.uint8)