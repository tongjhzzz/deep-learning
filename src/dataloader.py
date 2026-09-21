"""
Day 7 · torchvision 数据集 + DataLoader
计划：写 src/dataloader.py，用 CIFAR10 或 MNIST + DataLoader(batch_size=64, shuffle=True)
验收：能打印出一个 batch 的形状（如 [64, 3, 32, 32]）

在项目根目录运行：python src/dataloader.py
"""

# TODO: 按计划写你的代码


import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

writer = SummaryWriter('logs/dataloader')
for epoch in range(1):
    step = 0
    for data in test_loader:
        imgs, targets = data
        print(imgs.shape)
        writer.add_images('Epoch: {}'.format(epoch), imgs, step)
        step += 1

writer.close()
