

from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img_path = "/Users/daxian/deep-learning/dataset/train/bees_image/154600396_53e1252e52.jpg"
writer = SummaryWriter('logs')
img = Image.open(img_path)
print(img)# size = 500 * 356
tensor_img = transforms.ToTensor()(img)
writer.add_image("changeing_image", tensor_img, 0)

trans_compose = transforms.Compose([transforms.Resize(200), transforms.ToTensor()])
tensor_img_2 = trans_compose(img)
writer.add_image("changeing_image", tensor_img_2, 1)

trans_compose_2 = transforms.Compose([transforms.Resize((400, 300)), transforms.ToTensor()])
tensor_img_3 = trans_compose_2(img)
writer.add_image("changeing_image", tensor_img_3, 2)

writer.close()


