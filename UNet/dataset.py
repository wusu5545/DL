import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))

        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L')) # Gray

        if self.transform is not None:
            transformation = self.transform(image = image, mask = mask)
            image = transformation['image']
            mask = transformation['mask']
            mask[mask == 255] = 1.

        return image, mask