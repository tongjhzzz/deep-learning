"""
Day 11 · 搭建小实战和 Sequential
计划：写 src/nn_seq.py 搭一个 CIFAR10 小网络
验收：打印的网络结构与手算一致

在项目根目录运行：python src/nn_seq.py
"""


from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter


class Web(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

web = Web()
print(web)
input = torch.ones((64, 3, 32, 32))
output = web(input)
print(output.shape)

writer = SummaryWriter("logs/nn_seq")
writer.add_graph(web, input)
writer.close()