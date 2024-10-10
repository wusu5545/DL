from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):

    def __init__(self, data, mode):
        self.data = data
        self.mode = mode

        self.train_transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            #tv.transforms.RandomHorizontalFlip(0.2),
            #tv.transforms.RandomVerticalFlip(0.2),
            #tv.transforms.RandomAffine(degrees=(-2, 2), translate=(0.02,0.02)),
            #tv.transforms.RandomResizedCrop((300, 300), scale=(0.98, 1.0), ratio=(1, 1)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=train_mean, std=train_std),
        ])

        self.val_transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=train_mean, std=train_std),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data.iloc[index, 0]
        image = imread(img_name)
        image = gray2rgb(image)
        image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        if self.mode == 'train':
            image = self.train_transform(image)
        if self.mode == 'val':
            image = self.val_transform(image)
        label = torch.tensor(np.array(self.data.iloc[index, 1:], dtype='float'))
        return image, label
