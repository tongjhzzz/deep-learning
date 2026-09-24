"""
Day 9 · 最大池化
计划：写 src/nn_maxpool.py，每层打印输入输出维度
验收：能自己手算出池化后的张量形状

在项目根目录运行：python src/nn_maxpool.py
"""

# TODO: 按计划写你的代码



import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch import nn


dataset = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset, batch_size=64)

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool(input)
        return output

writer = SummaryWriter("logs/nn_maxpool")
model = Model()

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("imput", imgs, step)
    output = model(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
