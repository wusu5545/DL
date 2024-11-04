import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt

def get_loaders(train_img_dir, train_mask_dir, val_img_dir,val_mask_dir,
                train_transform, val_transform, batch_size, num_workers, pin_memory = True):

    train_dataset = CarvanaDataset(train_img_dir, train_mask_dir, train_transform)
    val_dataset = CarvanaDataset(val_img_dir, val_mask_dir, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader