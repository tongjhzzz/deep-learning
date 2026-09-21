# PyTorch 环境自检脚本
# 运行方式：cd ~/deep-learning && uv run python test_torch.py
import torch
import torchaudio
import torchvision

print("torch 版本:", torch.__version__)
print("torchvision 版本:", torchvision.__version__)
print("torchaudio 版本:", torchaudio.__version__)

# 1. CPU 上的张量运算（矩阵乘法）
x = torch.randn(3, 3)
y = torch.randn(3, 3)
z = x @ y
print("CPU 矩阵乘法结果形状:", tuple(z.shape))

# 2. 自动求导（深度学习训练的核心机制）
a = torch.tensor(2.0, requires_grad=True)
b = a**2 + 3 * a
b.backward()
print("自动求导检查: b 对 a 的导数 =", a.grad.item(), "(期望 7.0)")

# 3. MPS（Apple 芯片的 GPU 加速）
if torch.backends.mps.is_available():
    xm = torch.randn(3, 3, device="mps")
    zm = xm @ xm
    print("MPS 加速可用，GPU 矩阵乘法结果形状:", tuple(zm.shape))
else:
    print("MPS 当前不可用（不影响学习，PyTorch 会用 CPU 正常运行）")

print("测试通过 ✓")
