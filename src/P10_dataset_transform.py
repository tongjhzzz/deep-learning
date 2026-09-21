

import torchvision 
from torch.utils.tensorboard import SummaryWriter

dataset_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10("data", train=True, transform=dataset_trans, download=True)
test_set = torchvision.datasets.CIFAR10("data", train=True, transform=dataset_trans, download=True)

writer = SummaryWriter("logs")
for i in range(20):
    img, target = test_set[i]
    writer.add_image("test_img", img, i)

writer.close()
