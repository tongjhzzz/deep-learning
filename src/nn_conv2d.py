"""
Day 9 · 卷积层
计划：写 src/nn_conv2d.py，每层打印输入输出维度
验收：能自己手算出 CIFAR10 图片经过这层网络后的张量形状

在项目根目录运行：python src/nn_conv2d.py
"""

# TODO: 按计划写你的代码

import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.conv2 = Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        self.x1 = self.conv1(x)
        self.x2 = self.conv2(self.x1)
        return self.x2

writer = SummaryWriter("logs/nn_conv2d_2")
model = Model()

step = 0
for data in dataloader:
    imgs, targets = data
    output = model(imgs)
    print(imgs.shape)
    print(output.shape)

    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()