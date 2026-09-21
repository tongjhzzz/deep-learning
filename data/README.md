# data/

这个目录留给**需要下载**的数据集，目前是空的。

## 为什么是空的

蚂蚁/蜜蜂数据集已经手动放在项目的 `dataset/` 里了，**不挪过来**，因为：

- `src/read_data.py`、`src/P9_transforms.py` 和 day3/day4/day5 三个 notebook 里有 6 处写死的
  `deep-learning/dataset` 路径，挪动会全部失效；
- `dataset/` 是课程（小土堆）和 torchvision 的惯用名字。

所以 `dataset/` 是「已有数据」，`data/` 是「待下载数据」，各管一摊。

## 谁会用到这里

Day 7 的 `src/dataloader.py`：

```python
from torchvision import datasets

train_set = datasets.CIFAR10(root="./data", train=True, download=True)
```

`root="./data"` 就会下载到 `data/cifar-10-batches-py/`。
MNIST 同理，落在 `data/MNIST/`。

下载慢的话，按计划里的提示可以先用小数据集把流程跑通。
