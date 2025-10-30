# 保存为 check_mps.py 然后 python check_mps.py 运行
import torch
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
x = torch.randn(1024, 1024, device="mps")
y = x @ x.t()
print("OK on:", y.device)
