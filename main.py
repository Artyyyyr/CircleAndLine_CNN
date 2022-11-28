import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import long
from torch.optim import SGD
import numpy as np

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=50, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(kernel_size=(10, 10), stride=(10, 10))
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(in_features=900, out_features=300)
        self.l2 = nn.Linear(in_features=300, out_features=100)
        self.l3 = nn.Linear(in_features=100, out_features=2)

    def forward(self, x):
        x = np.array(x).reshape(1, 3, 647, 431)
        x = torch.tensor(x, dtype=torch.float32).cuda()
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(900)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)

        return x


class Data_img(Dataset):
    def __init__(self):
        self.x_im = []
        self.y = []
        for img in os.listdir("train_lin"):
            self.x_im.append(torch.tensor(Image.open("train_lin/" + img).resize((647, 431)).getdata()))
            self.y.append(1)
            print(img + " is loaded")
        for img in os.listdir("train_ron"):
            self.x_im.append(torch.tensor(Image.open("train_ron/" + img).resize((647, 431)).getdata()))
            self.y.append(0)
            print(img + " is loaded")

    def __getitem__(self, item):
        return self.x_im[item], self.y[item]

    def __len__(self):
        return len(self.x_im)


def test():
    ron_t = 0
    ron_f = 0
    lin_t = 0
    lin_f = 0
    for img in os.listdir("test/line"):
        data = torch.tensor(Image.open("test/line/" + img).resize((647, 431)).getdata())
        res = net(data)
        if res[0] > res[1]:
            lin_f = lin_f + 1
            print(img + " is false")
        else:
            lin_t = lin_t + 1
            print(img + " is true")
    for img in os.listdir("test/round"):
        data = torch.tensor(Image.open("test/round/" + img).resize((647, 431)).getdata())
        res = net(data)
        if res[1] > res[0]:
            ron_f = ron_f + 1
            print(img + " is false")
        else:
            ron_t = ron_t + 1
            print(img + " is true")
    print("Line")
    print(str(lin_t / (lin_f + lin_t) * 100) + "% is true")
    print("true: " + str(lin_t))
    print("false: " + str(lin_f))
    print("Round")
    print(str(ron_t / (ron_f + ron_t) * 100) + "% is true")
    print("true: " + str(ron_t))
    print("false: " + str(ron_f))

net = torch.load("nets/net.pth").cuda()
test()
Image.open("test/round/test_round (1).bmp").resize((671, 431)).show()
"""
dataset = Data_img()
data_loaded = DataLoader(dataset=dataset, batch_size=20, shuffle=True)
graph_loss_l = []
graph_loss_r = []
count = 0

e = 300
n = 1

op = torch.optim.SGD(params=net.parameters(), lr=0.00001)
for enoch in range(e):
    x, y = next(iter(data_loaded))
    for i in range(len(x)):
        op.zero_grad()
        loss = F.cross_entropy(net(x[i]), y[i])
        loss.backward()
        op.step()
    #if enoch % n == 1:
    print(str(loss.data) + "  " + str(y[9]))
    print("[" + str(count/(e/n)) + "]")
    if y[9] == 1:
        graph_loss_l.append(loss.detach().numpy())
    else:
        graph_loss_r.append(loss.detach().numpy())
    count = count + 1

plt.plot(np.arange(0, len(graph_loss_l)), graph_loss_l)
plt.plot(np.arange(0, len(graph_loss_r)), graph_loss_r)
plt.show()
torch.save(net, "nets/net.pth")
"""