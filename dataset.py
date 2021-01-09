import os
import sys
import numpy as np
import time
import cv2
from random import randint
from tqdm import tqdm
from PIL import Image,ImageOps,ImageEnhance

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import image

import torch
from torch.nn import Parameter
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset,DataLoader,Subset

import matplotlib.pyplot as plt
import h5py

def stringify(num, places):
    string = str(num)
    length = len(string)
    diff   = places - length
    zeros  = diff * str(0)
    return zeros + string

def normalize(x, verbose=False):
    x = (x-x.min())/(x.max()-x.min())
    x = x * 2
    x = x - 1
    if verbose:
        print('Normalized Min Max:', x.min(), x.max())
    return x

class Data:
    def __init__(self, dims=(64, 64), path=r'F:\William\dataset_HDF5', mode = 'train'):
        self.f = h5py.File(os.path.join(path, "LUNG_DB.hdf5"), 'r')
        self.g = self.f.get('scans')
        self.ids = list(self.g.keys())

    def get_ids(self):
        return self.ids
    
    def get_image(self, id):
        return self.g.get(id)

    def close(self):
        self.f.close()


class Dataset(Dataset):
    def __init__(self, dims=(64, 64), path=r'F:\William\dataset_HDF5', mode = 'train'):
        self.dims = dims
        self.f = h5py.File(os.path.join(path, "LUNG_DB.hdf5"), 'r')
        self.g = self.f.get('scans')
        self.ids = list(self.g.keys())
        train_ids, test_ids = train_test_split(self.ids, train_size=0.95, 
                                               test_size=0.05, random_state=86)
        if mode == 'train':
            self.ids = train_ids
            print('Using', len(self.ids), 'scans.')
        else:
            self.ids = test_ids
            print('Using', len(self.ids), 'scans.')
                    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self,idx):
        id = self.ids[idx]
        imgs = self.g.get(id)
        imgs = np.array(imgs)
        n_imgs = imgs.shape[0]
        n = randint(1, n_imgs-2)
        img = imgs[n] # Select 1 image
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        return img


if __name__ == "__main__":
    path = r'F:\William\dataset_HDF5'
    train_dataset = Dataset(path=path, mode='train')
    train_loader = DataLoader(train_dataset, 24, shuffle=True, num_workers=0, pin_memory=True)
    print('Number of batches:', len(train_loader))
    batch = iter(train_loader)
    x = next(batch)
    print(x.size(), x.dtype)
    print(x.min(), x.max())
    print()
    path = r'F:\William\dataset_HDF5'
    test_dataset = Dataset(path=path, mode='test')
    test_loader = DataLoader(test_dataset, 24, shuffle=True, num_workers=0, pin_memory=True)
    print('Number of batches:', len(test_loader))
    batch = iter(train_loader)
    x = next(batch)
    print(x.size(), x.dtype)
    print(x.min(), x.max())
    #f.close()



