from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

img_path = '/Users/daxian/deep-learning/dataset/train/bees_image/85112639_6e860b0469.jpg'
img = Image.open(img_path)

trans = transforms.ToTensor()
tensor_img = trans(img)

writer = SummaryWriter('logs')
writer.add_image("tensor_img", tensor_img)

writer.close()
