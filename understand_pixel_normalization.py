#!/usr/bin/env python3
"""
理解MAE中的像素归一化问题
解释归一化像素 vs 原始像素，以及正确的映射方法
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

# 解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import models_mae

def explain_pixel_normalization():
    """解释像素归一化的概念"""
    
    print("📚 像素归一化详解")
    print("=" * 60)
    
    print("\n🎯 两种训练目标的区别:")
    
    print("\n1️⃣ 原始像素 (norm_pix_loss=False):")
    print("   • 目标: 直接预测原始的RGB像素值")
    print("   • 范围: [0, 1] (ToTensor后)")
    print("   • 损失: MSE(预测像素, 原始像素)")
    print("   • 优点: 直观，容易理解")
    print("   • 缺点: 不同patch的像素值差异很大")
    
    print("\n2️⃣ 归一化像素 (norm_pix_loss=True):")
    print("   • 目标: 预测每个patch内归一化的像素值")
    print("   • 范围: 每个patch内均值为0，标准差为1")
    print("   • 损失: MSE(预测像素, 归一化像素)")
    print("   • 优点: 消除了不同patch间的亮度差异")
    print("   • 缺点: 需要正确的反归一化才能可视化")

def demonstrate_normalization_difference():
    """演示归一化的具体差异"""
    print(f"\n🧪 演示归一化的具体差异...")
    
    # 创建一个测试图像
    test_img = torch.zeros(3, 224, 224)
    
    # 创建不同亮度的区域
    test_img[:, 0:112, 0:112] = 0.2    # 暗区域
    test_img[:, 0:112, 112:224] = 0.8  # 亮区域
    test_img[:, 112:224, 0:112] = 0.5  # 中等区域
    test_img[:, 112:224, 112:224] = 0.9 # 很亮区域
    
    print(f"测试图像范围: [{test_img.min():.3f}, {test_img.max():.3f}]")
    
    # 模拟patch处理 (简化版)
    patch_size = 16
    patches = []
    normalized_patches = []
    
    for i in range(0, 224, patch_size):
        for j in range(0, 224, patch_size):
            # 提取patch
            patch = test_img[:, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
            
            # 归一化patch (在每个patch内)
            patch_flat = patch.flatten()
            if patch_flat.std() > 1e-6:  # 避免除零
                normalized_patch = (patch_flat - patch_flat.mean()) / patch_flat.std()
            else:
                normalized_patch = patch_flat - patch_flat.mean()
            
            normalized_patches.append(normalized_patch.reshape(patch.shape))
    
    # 显示几个patch的差异
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(4):
        patch_idx = i * 30  # 选择不同的patch
        if patch_idx < len(patches):
            original_patch = patches[patch_idx]
            normalized_patch = normalized_patches[patch_idx]
            
            # 显示原始patch
            axes[0, i].imshow(original_patch.permute(1, 2, 0))
            axes[0, i].set_title(f'Original Patch {i+1}\nMean: {original_patch.mean():.3f}')
            axes[0, i].axis('off')
            
            # 显示归一化patch (需要重新映射到[0,1]用于显示)
            norm_patch_display = (normalized_patch - normalized_patch.min()) / (normalized_patch.max() - normalized_patch.min() + 1e-6)
            axes[1, i].imshow(norm_patch_display.permute(1, 2, 0))
            axes[1, i].set_title(f'Normalized Patch {i+1}\nMean: {normalized_patch.mean():.3f}')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('patch_normalization_demo.png', dpi=150, bbox_inches='tight')
    print("✅ patch归一化演示保存: patch_normalization_demo.png")
    plt.close()

def test_correct_denormalization():
    """测试正确的反归一化方法"""
    print(f"\n🔧 测试正确的反归一化方法...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 加载使用norm_pix_loss=True训练的模型
    model = models_mae.mae_vit_base_patch16(norm_pix_loss=True)
    
    checkpoint_path = './output_image_repair_v1/checkpoint-19.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        print("✅ 加载norm_pix_loss=True模型")
    
    model.to(device)
    model.eval()
    
    # 加载测试图片
    try:
        ds = load_dataset("Mercity/AnimeDiffusion_Dataset")
        sample = ds['train'][0]
        original_img = sample['image']
        
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
    except Exception as e:
        print(f"使用默认图片: {e}")
        original_img = Image.new('RGB', (224, 224), color='red')
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(int(224 * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(original_img).unsqueeze(0).to(device)
    
    # MAE前向传播
    with torch.no_grad():
        loss, pred, mask = model(img_tensor, mask_ratio=0.25)
        
        print(f"预测值范围: [{pred.min():.3f}, {pred.max():.3f}]")
        
        # 方法1: 标准的unpatchify (模型内置)
        recon_standard = model.unpatchify(pred)
        print(f"标准重建范围: [{recon_standard.min():.3f}, {recon_standard.max():.3f}]")
        
        # 方法2: 手动处理归一化像素的反归一化
        recon_manual = manual_denormalize_patches(pred, img_tensor, mask, model)
        print(f"手动重建范围: [{recon_manual.min():.3f}, {recon_manual.max():.3f}]")
    
    # 显示不同方法的结果
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    original_display = torch.clamp(inv_normalize(img_tensor[0]).cpu(), 0, 1)
    recon_standard_display = torch.clamp(inv_normalize(recon_standard[0]).cpu(), 0, 1)
    recon_manual_display = torch.clamp(inv_normalize(recon_manual[0]).cpu(), 0, 1)
    
    # 创建掩码可视化
    mask_vis = mask.detach().unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
    mask_vis = model.unpatchify(mask_vis)
    masked_img = original_display * (1 - mask_vis[0].cpu()) + mask_vis[0].cpu() * 0.5
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(original_display.permute(1, 2, 0))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(masked_img.permute(1, 2, 0))
    axes[1].set_title('25% Masked')
    axes[1].axis('off')
    
    axes[2].imshow(recon_standard_display.permute(1, 2, 0))
    axes[2].set_title(f'Standard Method\nLoss: {loss.item():.3f}')
    axes[2].axis('off')
    
    axes[3].imshow(recon_manual_display.permute(1, 2, 0))
    axes[3].set_title('Manual Denormalization\n(Experimental)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('denormalization_methods_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ 反归一化方法对比保存: denormalization_methods_comparison.png")
    plt.close()

def manual_denormalize_patches(pred, original_img, mask, model):
    """手动实现归一化像素的反归一化"""
    
    # 这是一个实验性的方法，尝试正确处理归一化像素
    B, L, D = pred.shape
    
    # 获取patch信息
    patch_size = model.patch_embed.patch_size[0]
    num_patches_per_dim = int(L**0.5)
    
    # 重建图像
    reconstructed = torch.zeros(B, 3, 224, 224, device=pred.device)
    
    # 获取原始图像的patch用于计算统计信息
    original_patches = model.patchify(original_img)  # [B, L, patch_size^2 * 3]
    
    for i in range(L):
        if mask[0, i] == 1:  # 只处理被掩盖的patch
            # 获取预测的归一化像素
            pred_patch = pred[0, i]  # [D] where D = patch_size^2 * 3
            
            # 获取对应的原始patch统计信息
            original_patch = original_patches[0, i]
            patch_mean = original_patch.mean()
            patch_std = original_patch.std()
            
            # 反归一化: normalized_pixel * std + mean
            if patch_std > 1e-6:
                denorm_patch = pred_patch * patch_std + patch_mean
            else:
                denorm_patch = pred_patch + patch_mean
            
            # 将patch放回图像
            h = i // num_patches_per_dim
            w = i % num_patches_per_dim
            
            patch_img = denorm_patch.reshape(3, patch_size, patch_size)
            reconstructed[0, :, h*patch_size:(h+1)*patch_size, w*patch_size:(w+1)*patch_size] = patch_img
    
    # 对于未被掩盖的patch，直接使用原始像素
    for i in range(L):
        if mask[0, i] == 0:  # 未被掩盖的patch
            h = i // num_patches_per_dim
            w = i % num_patches_per_dim
            reconstructed[0, :, h*patch_size:(h+1)*patch_size, w*patch_size:(w+1)*patch_size] = \
                original_img[0, :, h*patch_size:(h+1)*patch_size, w*patch_size:(w+1)*patch_size]
    
    return reconstructed

def create_detailed_explanation():
    """创建详细的技术解释"""
    print(f"\n📖 创建详细技术解释...")
    
    explanation = """
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
"""
    
    with open('pixel_normalization_explanation.md', 'w') as f:
        f.write(explanation)
    
    print("✅ 技术解释保存: pixel_normalization_explanation.md")

def test_correct_reconstruction_method():
    """测试正确的重建方法"""
    print(f"\n🧪 测试正确的重建方法...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 创建一个自定义的MAE模型，实现正确的反归一化
    class FixedMAE(models_mae.MaskedAutoencoderViT):
        def forward_decoder(self, x, ids_restore):
            # 标准的decoder前向传播
            x = self.decoder_embed(x)
            
            # 添加mask tokens
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            x = torch.cat([x[:, :1, :], x_], dim=1)
            
            # 添加位置编码
            x = x + self.decoder_pos_embed
            
            # 应用decoder blocks
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)
            
            # 预测像素
            x = self.decoder_pred(x)
            x = x[:, 1:, :]  # 移除class token
            
            return x
        
        def unpatchify_corrected(self, x, original_img, mask):
            """正确的unpatchify，处理归一化像素"""
            B, L, D = x.shape
            p = self.patch_embed.patch_size[0]
            h = w = int(L**.5)
            
            x = x.reshape(shape=(B, h, w, p, p, 3))
            x = torch.einsum('nhwpqc->nchpwq', x)
            imgs = x.reshape(shape=(B, 3, h * p, w * p))
            
            if self.norm_pix_loss:
                # 如果使用归一化像素损失，需要正确的反归一化
                original_patches = self.patchify(original_img)
                
                for i in range(L):
                    if mask[0, i] == 1:  # 只处理被掩盖的patch
                        # 获取原始patch的统计信息
                        original_patch = original_patches[0, i]
                        patch_mean = original_patch.mean()
                        patch_std = original_patch.std()
                        
                        # 反归一化
                        h_idx = i // w
                        w_idx = i % w
                        
                        if patch_std > 1e-6:
                            imgs[0, :, h_idx*p:(h_idx+1)*p, w_idx*p:(w_idx+1)*p] = \
                                imgs[0, :, h_idx*p:(h_idx+1)*p, w_idx*p:(w_idx+1)*p] * patch_std + patch_mean
                        else:
                            imgs[0, :, h_idx*p:(h_idx+1)*p, w_idx*p:(w_idx+1)*p] = \
                                imgs[0, :, h_idx*p:(h_idx+1)*p, w_idx*p:(w_idx+1)*p] + patch_mean
            
            return imgs
    
    # 测试修正的方法
    fixed_model = FixedMAE(norm_pix_loss=True)
    fixed_model.load_state_dict(model.state_dict())
    fixed_model.to(device)
    fixed_model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(original_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        loss, pred, mask = fixed_model(img_tensor, mask_ratio=0.25)
        
        # 标准方法
        recon_standard = fixed_model.unpatchify(pred)
        
        # 修正方法
        recon_corrected = fixed_model.unpatchify_corrected(pred, img_tensor, mask)
    
    # 可视化对比
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    original_display = torch.clamp(inv_normalize(img_tensor[0]).cpu(), 0, 1)
    standard_display = torch.clamp(inv_normalize(recon_standard[0]).cpu(), 0, 1)
    corrected_display = torch.clamp(inv_normalize(recon_corrected[0]).cpu(), 0, 1)
    
    axes[0].imshow(original_display.permute(1, 2, 0))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(standard_display.permute(1, 2, 0))
    axes[1].set_title('Standard Unpatchify\n(Current Method)')
    axes[1].axis('off')
    
    axes[2].imshow(corrected_display.permute(1, 2, 0))
    axes[2].set_title('Corrected Unpatchify\n(Fixed Method)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('corrected_reconstruction_test.png', dpi=150, bbox_inches='tight')
    print("✅ 修正重建方法测试保存: corrected_reconstruction_test.png")
    plt.close()
    
    # 计算质量指标
    mse_standard = torch.mean((original_display - standard_display)**2).item()
    mse_corrected = torch.mean((original_display - corrected_display)**2).item()
    
    psnr_standard = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse_standard))).item()
    psnr_corrected = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse_corrected))).item()
    
    print(f"\n📊 方法对比:")
    print(f"  标准方法: PSNR {psnr_standard:.1f}dB, MSE {mse_standard:.4f}")
    print(f"  修正方法: PSNR {psnr_corrected:.1f}dB, MSE {mse_corrected:.4f}")
    print(f"  改进: {psnr_corrected - psnr_standard:+.1f}dB")

def main():
    """主函数"""
    print("📚 MAE像素归一化问题深度解析")
    print("=" * 50)
    
    # 1. 解释概念
    explain_pixel_normalization()
    
    # 2. 演示差异
    demonstrate_normalization_difference()
    
    # 3. 创建技术解释
    create_detailed_explanation()
    
    # 4. 测试正确的反归一化
    test_correct_reconstruction_method()
    
    print(f"\n🎯 总结:")
    print(f"  🔴 当前问题: norm_pix_loss=True时反归一化不正确")
    print(f"  🟡 临时解决: 使用norm_pix_loss=False")
    print(f"  🟢 根本解决: 实现正确的patch级反归一化")

if __name__ == "__main__":
    main()


