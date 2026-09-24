"""
Day 8 · 神经网络骨架 nn.Module
计划：写 src/nn_module.py 自定义网络
验收：forward 输出的张量形状与手算一致

在项目根目录运行：python src/nn_module.py
"""

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


model = Model()
x = torch.tensor(1.0)
output = model(x)
print(output)
