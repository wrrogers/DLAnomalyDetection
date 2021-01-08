import argparse
import sys
import os
import time
import warnings

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from vqvae import VQVAE
from scheduler import CycleScheduler

from dataset import Dataset

from parameters import Config


def get_sample(epoch, loader, model, device):

    img = next(iter(loader))
    
    model.eval()

    img = img.to(device)

    sample = img[:args.data.sample_size]

    with torch.no_grad():
        out, _ = model(sample)

    ts = time.time()

    utils.save_image(
        torch.cat([sample, out], 0),        
        'sample/{}.png'.format(ts),
        nrow=args.data.sample_size,
        normalize=True,
        range=(-1, 1),
    )


def main(args):
    dataset = Dataset(path=args.data.path, mode='train')
    loader = DataLoader(dataset, args.data.sample_size, shuffle=True, num_workers=0, pin_memory=True)
    
    model = VQVAE().to(args.hardware.sample_device)
    model.load_state_dict(torch.load('checkpoint/vqvae_007.pt'))  
    
    sample = get_sample(1, loader, model, args.hardware.sample_device)



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    args = Config()

    print('\n', args, '\n')

    main(args)
