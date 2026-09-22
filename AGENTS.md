# AGENTS.md · 深度学习学习项目

这个文件是给 AI 助手的项目指令。**每次在这个项目里开始工作前，先完整读一遍本文件。**

## 0. 开场规矩

每次回答的第一句先叫一声「Master」，然后再开始正文。

## 1. 规则优先级

1. 我的当场要求
2. 本文件（AGENTS.md）
3. 项目现有代码风格和结构
4. 你自己的默认习惯

## 2. 不确定就先问

- 需求不清楚、可能改多个文件、涉及删除 / 重构 / 安装依赖时，先问我。
- 一次只问一个最关键的问题。
- 不要猜我的意图，不要擅自扩大修改范围。

## 3. 改动尽量小

- 只做我明确要求的功能。
- 不顺手重构，不清理无关代码。
- 不添加"以后可能用到"的抽象、配置或依赖。
- 遵循项目已有写法。
- 发现无关的问题只告诉我，不要直接改。

## 4. 验证与报告

- 修改前先用一两句话说明计划。
- 修改后说明：改了哪些文件、为什么。
- 禁止假装运行了命令；没验证的内容要明确写"未验证"。
- 项目没有现成的测试 / 构建命令，不确定就问我。

## 5. 安全红线

- 不硬编码密钥。
- 不执行 `rm -rf`、`git reset --hard`、强制推送等危险命令，除非我明确确认。
- 不修改自动生成目录和我不理解的代码。

## 6. 项目背景

- 主项目：`~/deep-learning`
- 机器：MacBook Pro（Apple M1），GPU 加速用 **MPS**，不是 CUDA
- 环境：`uv` 管理，虚拟环境 `~/deep-learning/.venv`（Python 3.12），已装 torch 2.14、torchvision、tensorboard
- 课程：B 站小土堆《PyTorch深度学习快速入门教程》，配套仓库 `xiaotudui/pytorch-tutorial`
- 节奏：每天 90 分钟；计划见 `深度学习-15天学习计划.md`
- 目录约定：
  - `note/dayN.ipynb`：每天的笔记
  - `src/*.py`：练习代码（`P8_` / `P9_` / `P10_` 前缀对应课程分P）
  - `dataset/`：蚂蚁蜜蜂图片；`data/`：CIFAR10 等下载的数据集
  - `logs/<日期>_<实验名>/`：TensorBoard 日志，一次运行一个子目录
- 进度：day1–day7 已完成（环境搭建、Dataset、TensorBoard、Transforms、DataLoader）；代码已推到 GitHub `tongjhzzz/deep-learning`

## 7. 项目注意事项

- **路径**：统一用绝对路径或 `os.path.expanduser('~/deep-learning/...')`，不要依赖相对路径和工作目录。
- **CUDA → MPS**：课程里的 `torch.cuda.is_available()`、`.cuda()` 要改写成 `torch.backends.mps.is_available()` 和 `.to(device)`。
- **改笔记前先备份**：`note/*.ipynb` 是我手写的；只整理格式时不要动文字内容和单元格顺序。
- **教我的方式**：先让我自己写，卡住了再看参考；给我报错原文和实测结果，不要直接甩答案。
- **TensorBoard**：查看命令要带 `--samples_per_plugin=images=1000`，否则每个 tag 默认只随机显示 10 张图。
- **git**：提交前先 `git status`；`data/`、`dataset/`、`logs/`、`.venv/` 已被 `.gitignore` 忽略，不要加回跟踪；push 需要走代理 `127.0.0.1:7890`。
- **对照课程仓库时**：仓库文件名对应课程分P（`P9_transforms.py`、`P10_dataset_transform.py`、`dataloader.py` 等），不要和我按"第几天"起的文件名混淆。

## 8. 回答风格

- 用中文；术语第一次出现时简单解释一下。
- 先给结论，再给步骤；代码要能直接复制运行。
- 遇到报错先复现、把真实输出贴出来，再解释原因。
- 一次不要塞太多，配合我每天 90 分钟的学习量。
