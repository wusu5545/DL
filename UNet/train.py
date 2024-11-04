import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A # augmented lib
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import Unet
from utils import get_loaders

import numpy as np
import random

# Hyperparameters

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCH = 2
NUM_WORKERS = 2
PIN_MEMORY = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

TRAIN_IMG_DIR = "./data/train_images"
TRAIN_MASK_DIR = "./data/train_masks"
VAL_IMG_DIR = "./data/val_images"
VAL_MASK_DIR = "./data/val_masks"

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240

train_loss = []
val_acc = []
val_dice = []

# set random seed
seed = random.randint(1, 100)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def trainer(loader, model, loss_fn, optimizer, scaler):
    loop = tqdm(loader)
    total_loss = 0.0
    for idx, (data, target) in enumerate(loop):
        data = data.to(DEVICE)
        target = target.unsqueeze(1).float().to(DEVICE)

        with torch.amp.autocast('cuda'):
            predict = model(data)
            loss = loss_fn(predict, target)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.unsqueeze(1).to(DEVICE)
            predictions = torch.sigmoid(model(x))
            predictions = (predictions > 0.5).float()
            num_correct += (predictions == y).sum()
            num_pixels += torch.numel(predictions)
            dice_score += (2 * (predictions * y).sum()
                           ) / (2 * (predictions * y).sum() + ((predictions * y) < 1).sum())

    accuracy = round(float(num_correct / num_pixels), 4)
    dice = round(float(dice_score / len(loader)), 4)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100 :.2f}")
    print(f"Dice Score: {dice_score / len(loader)}")

    model.train()
    return accuracy, dice


def main():
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit = 35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean = [0.0, 0.0, 0.0],
            std = [1.0, 1.0, 1.0],
            max_pixel_value = 255.0,
        ),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean = [0.0, 0.0, 0.0],
            std = [1.0, 1.0, 1.0],
            max_pixel_value = 255.0,
        ),
        ToTensorV2(),
    ])

    train_loader, val_loader = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR,
                                           train_transform, val_transform, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)

    model = Unet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(NUM_EPOCH):
        print("Current Epoch: ", epoch)
        loss = trainer(train_loader, model, loss_fn, optimizer, scaler)
        train_loss.append(loss)

        accuracy, dice = check_accuracy(val_loader, model)
        val_acc.append(accuracy)
        val_dice.append(dice)

if __name__ == "__main__":
    main()