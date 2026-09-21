# src/

按 `深度学习-15天学习计划.md` 的每日交付物组织。文件命名与计划书一一对应。

## 运行约定

**所有脚本都在项目根目录运行**，这样 `SummaryWriter("logs")` 才会写到根目录的 `logs/`：

```bash
cd ~/deep-learning
python src/P8_Tensorboard.py
tensorboard --logdir=logs
```

如果在 `src/` 里面运行（`cd src && python P8_Tensorboard.py`），日志会落到 `src/logs/`，
TensorBoard 就找不到根目录的记录了。

## 进度对照

| Day | 文件 | 状态 |
| --- | --- | --- |
| 1 | `../test_torch.py`（按计划留在根目录） | ✅ 环境自检通过 |
| 2 | —（见 `note/day2.ipynb`） | ✅ 法宝函数 |
| 3 | `read_data.py` | ✅ 原根目录同名文件迁入 |
| 4 | `P8_Tensorboard.py` | ✅ 由 `note/day4.ipynb` 已跑通代码整理 |
| 5 | `P9_transforms.py` | ✅ 原名 `transforms_learning.py` |
| 6 | `P10_dataset_transform.py` | ⬜ 待写 |
| 7 | `dataloader.py` | ⬜ 待写（数据集下载到 `../data/`） |
| 8 | `nn_module.py`、`nn_conv.py`（选做） | ⬜ 待写 |
| 9 | `nn_conv2d.py`、`nn_maxpool.py` | ⬜ 待写 |
| 10 | `nn_relu.py`、`nn_linear.py` | ⬜ 待写 |
| 11 | `nn_seq.py`、`nn_loss.py` | ⬜ 待写 |
| 12 | `nn_optim.py`、`model_pretrained.py` | ⬜ 待写 |
| 13 | `model_save.py`、`model_load.py`、`train.py` | ⬜ 待写 |
| 14 | `train-cpu.py`、`train_mps.py` | ⬜ 待写 |
| 15 | `test.py` | ⬜ 待写 |

每个待写的空文件头部都写了当天的**计划内容**和**验收标准**，直接照着填即可。

## 已知待修

- `read_data.py` 和 `P8_Tensorboard.py` 里的数据集路径是写死的绝对路径
  （`/Users/daxian/deep-learning/dataset/...`）。等 day13 写 `train.py` 前建议统一改成相对路径。
- `read_data.py` 末尾的 `img.show()` 在模块顶层，一运行就弹窗阻塞，且没有
  `if __name__ == "__main__":` 保护（这是课程原样写法，不是错误，但自己写训练脚本时别照抄）。
