import os
import numpy as np
import tarfile
from PIL import Image
from tqdm import tqdm
import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


class MVTecDataset(Dataset):
    def __init__(self, root_path='E:/', class_name='bottle', resize=256, cropsize=224):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.root_path = root_path
        self.class_name = class_name
        self.resize = resize
        self.cropsize = cropsize
        self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # download dataset if not exist
        #self.download()

        # load dataset
        self.x = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])
    
        self.file_list = None

    def __getitem__(self, idx):
        x = self.x[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        return x, idx

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        
        img_dir = os.path.join(self.mvtec_folder_path, self.class_name)

        x = []

        folders = ['train', 'test']
        
        for folder in folders:

            img_types = sorted(os.listdir(os.path.join(img_dir, folder)))
    
            #print(img_types)
            
            for img_type in img_types:
    
                # load images
                img_type_dir = os.path.join(img_dir, folder, img_type)
                if not os.path.isdir(img_type_dir):
                    continue
                img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                         for f in os.listdir(img_type_dir)
                                         if f.endswith('.png')])
                x.extend(img_fpath_list)

        #self.file_list = x

        return list(x)
    
if __name__ == '__main__':
    train_dataset = MVTecDataset(class_name='bottle')
    train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True, shuffle=False)
    
    print(len(train_dataloader))
    
    x = next(iter(train_dataloader))
    
    print(x.size())
    
    img1 = np.moveaxis(x[0].detach().cpu().numpy(), 0, -1)
    img1 = (img1 - np.min(img1))/np.ptp(img1)
    
    plt.imshow(img1)
    plt.show()
    
    img2 = np.moveaxis(x[0].detach().cpu().numpy(), 0, -1)
    img2 = (img2 - np.min(img2))/np.ptp(img2)
    
    plt.imshow(img2)
    plt.show()
