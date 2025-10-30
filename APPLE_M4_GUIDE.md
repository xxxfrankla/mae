# MAE 在 Apple M4 24GB 上的使用指南

## 🎉 测试结果总结

你的 **Apple M4 24GB** 配置**完全支持**运行 MAE (Masked Autoencoders) 项目！

### ✅ 验证通过的功能

- **PyTorch MPS 支持**: 完全兼容，可以利用 GPU 加速
- **MAE ViT-Base**: 111.9M 参数，运行正常
- **MAE ViT-Large**: 329.5M 参数，运行正常  
- **ViT 分类模型**: 86.6M 参数，运行正常
- **内存管理**: 支持 batch_size 1-8，内存使用良好

## 🛠️ 环境设置

### 1. 创建 Conda 环境
```bash
conda create -n mae-mps python=3.11
conda activate mae-mps
```

### 2. 安装依赖
```bash
# 安装 PyTorch (支持 MPS)
pip install torch torchvision torchaudio

# 安装 MAE 特定依赖
pip install timm==0.3.2

# 安装可视化依赖
pip install jupyter matplotlib
```

### 3. 修复兼容性问题
已自动修复 `timm==0.3.2` 与 PyTorch 2.9.0 的兼容性问题。

### 4. 解决 OpenMP 冲突
每次运行前执行：
```bash
source setup_env.sh
```
或者手动设置：
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

## 🚀 快速开始

### 1. 环境测试
```bash
python test_mae_m4.py
```

### 2. 运行可视化演示
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
jupyter notebook demo/mae_visualize.ipynb
```

### 3. 模型推理测试
```python
import torch
import models_mae

# 设置设备
device = torch.device('mps')

# 加载模型
model = models_mae.mae_vit_base_patch16()
model = model.to(device)

# 测试推理
x = torch.randn(1, 3, 224, 224, device=device)
with torch.no_grad():
    loss, pred, mask = model(x, mask_ratio=0.75)

print(f"Loss: {loss.item():.4f}")
print(f"Prediction shape: {pred.shape}")
```

## 📊 性能建议

### 内存优化
- **推荐 batch_size**: 4-8 (ViT-Base), 2-4 (ViT-Large)
- **最大 batch_size**: 8 (已测试通过)
- **内存清理**: 使用 `torch.mps.empty_cache()` 释放显存

### 模型选择
- **学习/实验**: ViT-Base (111.9M 参数)
- **高性能需求**: ViT-Large (329.5M 参数)
- **避免使用**: ViT-Huge (参数过大，可能内存不足)

## 🎯 实际应用场景

### 1. 模型评估
```bash
# 下载预训练模型
wget https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth

# 评估模型 (需要 ImageNet 数据集)
export KMP_DUPLICATE_LIB_OK=TRUE
python main_finetune.py --eval --resume mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 8 --data_path ${IMAGENET_DIR}
```

### 2. 小规模微调
```bash
# 微调预训练模型
export KMP_DUPLICATE_LIB_OK=TRUE
python main_finetune.py \
    --batch_size 8 \
    --model vit_base_patch16 \
    --finetune mae_pretrain_vit_base.pth \
    --epochs 50 \
    --blr 5e-4 \
    --data_path ${YOUR_DATASET}
```

### 3. 可视化分析
- 运行 Jupyter notebook 查看 MAE 的掩盖和重建过程
- 分析不同掩盖比例的效果
- 可视化学习到的特征表示

## ⚠️ 注意事项

### 限制
1. **完整预训练**: 原始设置需要 64 个 V100 GPU，你的设备无法进行完整规模的预训练
2. **大规模数据集**: ImageNet 等大型数据集需要足够的存储空间
3. **中文字体**: matplotlib 可能无法显示中文，但不影响功能

### 解决方案
1. **使用预训练模型**: 下载官方预训练权重进行微调
2. **小规模实验**: 使用较小的数据集进行概念验证
3. **渐进式学习**: 从 ViT-Base 开始，逐步尝试更大的模型

## 📈 性能对比

| 模型 | 参数量 | 内存使用 | 推荐 Batch Size | 适用场景 |
|------|--------|----------|----------------|----------|
| ViT-Base | 111.9M | 低 | 4-8 | 学习、实验 |
| ViT-Large | 329.5M | 中 | 2-4 | 高性能应用 |
| ViT-Huge | 632M+ | 高 | 1-2 | 谨慎使用 |

## 🔗 有用链接

- [原始论文](https://arxiv.org/abs/2111.06377)
- [预训练模型下载](https://github.com/facebookresearch/mae#fine-tuning-with-pre-trained-checkpoints)
- [可视化演示](https://colab.research.google.com/github/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb)

## 🎊 结论

你的 Apple M4 24GB 配置非常适合：
- ✅ MAE 模型学习和实验
- ✅ 预训练模型的微调
- ✅ 小到中等规模的研究项目
- ✅ 可视化和分析工作

虽然无法进行完整规模的预训练，但对于大多数研究和应用场景来说，这个配置已经非常强大了！
