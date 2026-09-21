"""
Day 4 · TensorBoard
计划：写 src/P8_Tensorboard.py，练习 add_scalar 和 add_image
验收：终端运行 tensorboard --logdir=logs，浏览器能看到曲线和图片

在项目根目录运行（cwd 必须是项目根目录，logs/ 才会落在根目录）：
    python src/P8_Tensorboard.py
    tensorboard --logdir=logs

说明：本文件是把 note/day4.ipynb 里已跑通的代码整理成的脚本版；
      笔记里 markdown 的进阶示例（tensor CHW / add_images 批量）没有落成代码。
"""
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# ---------- add_scalar：画 y = x 的曲线 ----------
writer = SummaryWriter("logs")
for i in range(100):
    writer.add_scalar("y = x", i, i)  # y轴在前，x轴在后
writer.close()

# ---------- add_image：numpy 的 HWC 数组，必须写 dataformats ----------
writer = SummaryWriter("logs")
img_path = "/Users/daxian/deep-learning/dataset/train/ants_image/5650366_e22b7e1065.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)

writer.add_image("test_image", img_array, 1, dataformats="HWC")
writer.close()
