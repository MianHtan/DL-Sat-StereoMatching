import torch
import torch.utils.data as data
from torchvision import transforms

import numpy as np
import logging
import os

from pathlib import Path
from glob import glob

from .read_data import *


class StereoDataset(data.Dataset):
    def __init__(self, resize, disp_reader, min_disp, max_disp):    

        self.is_test = False
        self.init_seed = False
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []
        self.resize = resize
        self.dispreader = disp_reader
        self.min_disp = min_disp
        self.max_disp = max_disp

    def __getitem__(self, index):

        index = index % len(self.image_list)
        
        img1 = read_img(self.image_list[index][0], self.resize)
        img2 = read_img(self.image_list[index][1], self.resize)
        
        img1 = img_norm(img1)
        img2 = img_norm(img2)

        if self.is_test:
            return img1, img2
        
        disp, valid = self.dispreader(self.disparity_list[index], self.resize, self.min_disp, self.max_disp)
        disp = torch.from_numpy(disp).float()
        valid = torch.from_numpy(valid).bool()

        return img1, img2, disp, valid
    
    def __len__(self):
        return len(self.image_list)
    
class DFC2019(StereoDataset):
    def __init__(self, root, resize, min_disp, max_disp, image_set='training'):
        super().__init__(resize=resize, disp_reader=read_disp_dfc, min_disp=min_disp, max_disp=max_disp)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, 'Track2-RGB-*/*LEFT_RGB.tif')))
        image2_list = sorted(glob(os.path.join(root, 'Track2-RGB-*/*RIGHT_RGB.tif')))
        disp_list = sorted(glob(os.path.join(root, 'Track2-Truth/*DSP.tif')))
        if image_set == "testing":
            image1_list = image1_list[::20]
            image2_list = image2_list[::20]
            disp_list = disp_list[::20]
        if image_set == "training":
            image1_list_test = image1_list[::20]
            image2_list_test = image2_list[::20]
            disp_list_test = disp_list[::20]
            image1_list = sorted(list(set(image1_list).difference(set(list(image1_list_test)))))
            image2_list = sorted(list(set(image2_list).difference(set(list(image2_list_test)))))
            disp_list = sorted(list(set(disp_list).difference(set(list(disp_list_test)))))
        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]

class WHUStereo(StereoDataset):
    def __init__(self, root, resize, min_disp, max_disp, image_set='training'):
        super().__init__(resize=resize, disp_reader=read_disp_whu, min_disp=min_disp, max_disp=max_disp)
        assert os.path.exists(root)

        if image_set == "training":
            image1_list = sorted(glob(os.path.join(root, 'train/left/*.tiff')))
            image2_list = sorted(glob(os.path.join(root, 'train/right/*.tiff')))
            disp_list = sorted(glob(os.path.join(root, 'train/disp/*.tiff')))

        if image_set == "testing":
            image1_list = sorted(glob(os.path.join(root, 'val/left/*.tiff')))
            image2_list = sorted(glob(os.path.join(root, 'val/right/*.tiff')))
            disp_list = sorted(glob(os.path.join(root, 'val/disp/*.tiff')))

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


def fetch_dataset(dataset_name, root, batch_size, resize, min_disp, max_disp, mode="training", pin_memory=False, shuffle=True, drop_last=True):
    
    if dataset_name == 'DFC2019':
        if mode == 'training':
            dataset = DFC2019(root=root, resize = resize, min_disp = min_disp, max_disp = max_disp, 
                            image_set='training')
        elif mode == 'testing':
            dataset = DFC2019(root=root, resize = resize, min_disp = min_disp, max_disp = max_disp,
                            image_set='testing')     
    elif dataset_name == 'WHUStereo':
        if mode == 'training':
            dataset = WHUStereo(root=root, resize = resize, min_disp = min_disp, max_disp = max_disp,
                            image_set='training')
        elif mode == 'testing':
            dataset = WHUStereo(root=root, resize = resize, min_disp = min_disp, max_disp = max_disp,
                            image_set='testing') 
    elif dataset_name == "all":
        if mode == 'training':
            dataset = DFC2019(root= '/media/win_d/honghao/training_data/DFC2019/track2_grayscale', resize = resize, min_disp = min_disp, max_disp = max_disp, image_set='training')
            dataset += WHUStereo(root='/media/win_d/honghao/training_data/WHUStereo/WHUStereo_8UC3', resize = resize, min_disp = min_disp, max_disp = max_disp, image_set='training')
            dataset += WHUStereo(root='/media/win_d/honghao/training_data/WHUStereo/WHUStereo_8UC3', resize = resize, min_disp = min_disp, max_disp = max_disp, image_set='training')
            dataset += WHUStereo(root='/media/win_d/honghao/training_data/WHUStereo/WHUStereo_8UC3', resize = resize, min_disp = min_disp, max_disp = max_disp, image_set='training')
        elif mode == 'testing':
            dataset = DFC2019(root='/media/win_d/honghao/training_data/DFC2019/track2_grayscale', resize = resize, min_disp = min_disp, max_disp = max_disp, image_set='testing')
            dataset += WHUStereo(root='/media/win_d/honghao/training_data/WHUStereo/WHUStereo_8UC3', resize = resize, min_disp = min_disp, max_disp = max_disp, image_set='testing')         
    else: 
        print("no such a dataset")

    train_loader = data.DataLoader(dataset = dataset, batch_size=batch_size,
                                   pin_memory=pin_memory, shuffle=shuffle,
                                   num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6)) - 2, drop_last=drop_last)
    return train_loader