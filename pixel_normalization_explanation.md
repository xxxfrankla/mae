
# MAE 像素归一化技术解释

## 问题的根源

你观察到的重建模糊问题，核心在于 **norm_pix_loss** 参数的理解和处理。

### 1. 原始像素 (norm_pix_loss=False)

```python
# 训练目标：直接预测原始像素值
target = original_pixels  # 范围 [0, 1]
loss = MSE(predicted_pixels, target)
```

**特点**:
- ✅ 直观易懂
- ✅ 可视化简单
- ❌ 不同patch间亮度差异大，训练困难

### 2. 归一化像素 (norm_pix_loss=True)

```python
# 训练目标：预测每个patch内归一化的像素值
for each_patch:
    patch_mean = patch.mean()
    patch_std = patch.std()
    normalized_patch = (patch - patch_mean) / patch_std
    
target = normalized_patches  # 每个patch内均值≈0，标准差≈1
loss = MSE(predicted_normalized_pixels, target)
```

**特点**:
- ✅ 消除patch间亮度差异
- ✅ 训练更稳定
- ❌ 反归一化复杂
- ❌ 容易出现可视化错误

## 正确的反归一化方法

### 当前的错误做法
```python
# 错误：直接用全图的归一化参数
reconstructed = model.unpatchify(pred)  # 这里有问题！
display = inv_normalize(reconstructed)  # 错误的反归一化
```

### 正确的做法
```python
# 正确：需要用每个patch的统计信息反归一化
for each_masked_patch:
    # 1. 获取原始patch的统计信息
    original_patch_mean = original_patch.mean()
    original_patch_std = original_patch.std()
    
    # 2. 反归一化预测值
    denormalized_patch = pred_patch * original_patch_std + original_patch_mean
    
    # 3. 放回对应位置
    reconstructed[patch_position] = denormalized_patch
```

## 为什么会出现噪声

1. **统计信息丢失**: 模型预测的是归一化像素，但反归一化时用错了统计信息
2. **patch间不连续**: 每个patch独立归一化，边界可能不连续
3. **训练目标不匹配**: 模型学习的目标和显示时的处理不一致

## 解决方案

### 方案A: 使用原始像素训练 (推荐)
```bash
# 简单有效，直接预测原始像素
python main_pretrain_animediffusion.py --norm_pix_loss=False
```

### 方案B: 正确处理归一化像素
需要修改unpatchify函数，正确处理每个patch的反归一化。

### 方案C: 混合方法
在训练时使用归一化像素，但在推理时特殊处理反归一化。
