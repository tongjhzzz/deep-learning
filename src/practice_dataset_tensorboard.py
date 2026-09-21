from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('logs')

class Mydata(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        self.trans = transforms.ToTensor()

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        tensor_img = self.trans(img)
        return tensor_img

    def __len__(self):
        return len(self.img_path)

root_dir = '/Users/daxian/deep-learning/dataset/train'
ants_label_dir = 'ants_image'
ants_dataset = Mydata(root_dir, ants_label_dir)

n = len(ants_dataset)
for i in range(10):
    img = ants_dataset[i]
    writer.add_image('test_img', img, i)

writer.close()
    
