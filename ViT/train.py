import torch
import torch.nn as nn
from torch import optim
import timeit
from tqdm import tqdm
from utils import get_loaders
from model import ViT

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Hyper-Parameters

EPOCHS = 50

BATCH_SIZE = 16
TRAIN_DF_DIR = "./data/train.csv"
TEST_DF_DIR = "./data/test.csv"

# Model Parameters
IN_CHANNELS = 1
IMG_SIZE = 28
PATCH_SIZE = 4
EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
DROPOUT = 0.001

NUM_HEADS = 8
ACTIVATION = "gelu"
NUM_ENCODERS = 4
NUM_CLASSES = 10

LEARNING_RATE = 1e-4
ADAM_WEIGHT_DECAY = 0
ADAM_BEATS = (0.9, 0.999)

train_dataloader, val_dataloader, test_dataloader = get_loaders(TRAIN_DF_DIR, TEST_DF_DIR, batch_size=BATCH_SIZE)

model = ViT(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES, DROPOUT,
            NUM_HEADS, ACTIVATION, NUM_ENCODERS, NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), betas = ADAM_BEATS, lr = LEARNING_RATE, weight_decay = ADAM_WEIGHT_DECAY)

start = timeit.default_timer()

for epoch in tqdm(range(EPOCHS), position=0, leave=True):
    model.train()
    train_labels = []
    train_preds = []
    train_loss = 0

    for idx, img_label in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        img = img_label["image"].float().to(DEVICE)
        label = img_label["label"].type(torch.uint8).to(DEVICE)
        y_pred = model(img)
        y_pred_label = torch.argmax(y_pred, dim = 1)

        train_labels.extend(label.cpu().detach())
        train_preds.extend(y_pred_label.cpu().detach())

        loss = criterion(y_pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss / (idx + 1)

    model.eval()
    val_labels = []
    val_preds = []
    val_loss = 0

    with torch.no_grad():
        for idx, img_label in enumerate(tqdm(val_dataloader, position=0, leave=True)):
            img = img_label["image"].float().to(DEVICE)
            label = img_label["label"].type(torch.uint8).to(DEVICE)
            y_pred = model(img)
            y_pred_label = torch.argmax(y_pred, dim = 1)

            val_labels.extend(label.cpu().detach())
            val_preds.extend(y_pred_label.cpu().detach())

            loss = criterion(y_pred, label)
            val_loss += loss.item()

    val_loss = val_loss / (idx + 1)

    print("-" * 30)
    print(f"Train Loss Epoch {epoch+1} : {train_loss:.4f}")
    print(f"Val Loss Epoch {epoch+1} : {val_loss:.4f}")
    print(
        f"Train Acc Epoch {epoch+1} : {sum(1 for x, y in zip(train_preds, train_labels) if x == y) / len(train_labels):.4f}")
    print(
        f"Val Acc Epoch {epoch+1}: {sum(1 for x, y in zip(val_preds, val_labels) if x == y) / len(val_labels):.4f}")
    print("-" * 30)

stop = timeit.default_timer()
print(f"Training Time: {stop - start:.2f}s")