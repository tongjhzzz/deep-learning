"""P10 常见 transforms 示例：Resize / ToTensor / Normalize / Compose / 反归一化。

运行：
    cd ~/deep-learning
    uv run python src/P10_transforms_demo.py

查看结果（另开一个终端）：
    ~/deep-learning/.venv/bin/tensorboard --logdir=/Users/daxian/deep-learning/note/logs_transforms_demo
"""
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

IMG_PATH = '/Users/daxian/deep-learning/dataset/train/ants_image/0013035.jpg'
LOG_DIR = '/Users/daxian/deep-learning/note/logs_transforms_demo'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def show(name, t):
    """打印张量的形状、类型和值域。"""
    print(f"{name:32s} shape={tuple(t.shape)} dtype={t.dtype} "
          f"值域 {t.min():.4f} ~ {t.max():.4f}")


def main():
    img = Image.open(IMG_PATH)
    print(f"{'原图(PIL)':32s} size={img.size} mode={img.mode}\n")

    # 1) 单个变换：PIL -> 张量，HWC -> CHW，0~255 -> 0~1
    img_tensor = transforms.ToTensor()(img)
    show("ToTensor 之后", img_tensor)

    # 2) 组合变换：顺序不能错，Normalize 必须在 ToTensor 后面
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img_norm = trans(img)
    show("Compose(Resize+ToTensor+Norm)", img_norm)

    # 3) 反归一化：变回 0~1，否则送进 add_image 会发灰发黑
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img_show = (img_norm * std + mean).clamp(0, 1)
    show("反归一化之后", img_show)

    # 4) 写进 TensorBoard，四个阶段一起对比
    writer = SummaryWriter(LOG_DIR)
    writer.add_image('compare/1_原图_HWC', np.array(img), 0, dataformats='HWC')
    writer.add_image('compare/2_ToTensor', img_tensor, 1)
    writer.add_image('compare/3_Normalize_发灰', img_norm, 2)
    writer.add_image('compare/4_反归一化', img_show, 3)
    writer.close()

    print(f"\n已写入 TensorBoard：{LOG_DIR}")
    print("查看：~/deep-learning/.venv/bin/tensorboard --logdir=" + LOG_DIR)


if __name__ == '__main__':
    main()
