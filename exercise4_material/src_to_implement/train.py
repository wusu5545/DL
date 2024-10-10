import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import torchvision as tv

batch_size = 16
learning_rate = 1e-4
#momentum = 0.9
weight_decay = 1e-5
epochs = 20
early_stopping_patience = -1

pretrained = False #pretraned model resnet34

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data = pd.read_csv('data.csv', sep=';')
# 80% of the data will be used for training, 20% for testing
train_data, test_data = train_test_split(data, test_size=0.2)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_loader = t.utils.data.DataLoader(ChallengeDataset(train_data, "train"), batch_size = batch_size, shuffle=True)
val_loader = t.utils.data.DataLoader(ChallengeDataset(test_data, "val"), batch_size = batch_size)

# create an instance of our ResNet model

if pretrained:
    res_net = tv.models.resnet34(pretrained=True)
    res_net.fc = nn.Sequential(nn.Linear(512, 2), nn.Sigmoid())
else:
    res_net = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
crit = t.nn.BCELoss()
optimizer = t.optim.Adam(res_net.parameters(), lr=learning_rate, weight_decay = weight_decay)
scheduler = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.1)

trainer = Trainer(model = res_net,
                  crit = crit,
                  optim = optimizer,
                  scheduler = [scheduler],
                  train_dl = train_loader,
                  val_test_dl=val_loader,
                  early_stopping_patience=early_stopping_patience,
                  cuda = True)

# go, go, go... call fit on trainer
res = trainer.fit(epochs = epochs)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')