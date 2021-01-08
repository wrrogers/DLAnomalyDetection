import argparse
import sys
import os
import warnings

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler

from dataset import Dataset

from parameters import Config


def train(epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, img in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()
        
        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        mse_sum += recon_loss.item() * img.shape[0]
        mse_n += img.shape[0]

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                f'lr: {lr:.5f}'
            )
        )

        if i % 100 == 0:
            model.eval()

            sample = img[:sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            utils.save_image(
                torch.cat([sample, out], 0),
                f'sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            model.train()


def main(args):
    
    dataset = Dataset(path=args.data.path, mode='train')
    loader = DataLoader(dataset, 32, shuffle=True, num_workers=0, pin_memory=True)
    
    model = VQVAE().to(device)
    model.load_state_dict(torch.load('checkpoint/first_stop.pt'))
    
    #model = nn.parallel.DistributedDataParallel(model)
    #model = nn.parallel.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=args.optim.lr)
    scheduler = None
    if args.optim.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.optim.lr,
            n_iter=len(loader) * args.train.n_epochs,
            momentum=None,
            warmup_proportion=0.05,
        )
    
    for i in range(args.train.n_epochs):
        train(i, loader, model, optimizer, scheduler, args.hardware.sample_device)

        if i % args.train.log_iter == 0:
            torch.save(model.state_dict(), f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    args = Config()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.hardware.available_gpu
    print('Visible Device:', os.environ['CUDA_VISIBLE_DEVICES'])

    print('\n', args, '\n')

    main(args)
