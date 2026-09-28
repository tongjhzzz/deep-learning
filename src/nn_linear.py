"""
Day 10 · 线性层及其他层
计划：写 src/nn_linear.py
验收：能说清 Flatten 的作用

在项目根目录运行：python src/nn_linear.py
"""



import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Linear
import torchvision



dataset = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset=dataset, batch_size=64, drop_last=True)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = tudui(output)
    print(output.shape)
