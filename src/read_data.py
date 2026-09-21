from torch.utils.data import Dataset
from PIL import Image
import os

class Mydata(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir.replace('_image', '')
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = '/Users/daxian/deep-learning/dataset/train'
ants_label_dir = 'ants_image'
bees_label_dir = 'bees_image'

ants_dataset = Mydata(root_dir, ants_label_dir)
bees_dataset = Mydata(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset

img, label = train_dataset[0]
img.show()

