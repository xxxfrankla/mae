#!/usr/bin/env python3
"""
改进的MAE重建演示
尝试解决噪声问题，提供更好的重建质量
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import models_mae
from animediffusion_dataset_loader import create_animediffusion_dataloader
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_improved_mae_model():
    """创建改进的MAE模型"""
    print("\n🛠️ 创建改进的MAE模型...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 创建基础模型
    model = models_mae.mae_vit_base_patch16()
    
    # 加载编码器预训练权重
    pretrain_path = 'pretrained_models/mae_pretrain_vit_base.pth'
    if os.path.exists(pretrain_path):
        print("📥 加载编码器预训练权重...")
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        
        encoder_state_dict = {}
        for key, value in checkpoint['model'].items():
            if not key.startswith('decoder') and key != 'mask_token':
                encoder_state_dict[key] = value
        
        model.load_state_dict(encoder_state_dict, strict=False)
        print("✅ 编码器权重加载成功")
    
    # 改进解码器初始化
    print("🎨 改进解码器初始化...")
    
    # 1. 使用更小的初始化权重
    def init_decoder_weights(m):
        if isinstance(m, nn.Linear):
            # 使用更小的标准差
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    # 只对解码器部分应用新的初始化
    for name, module in model.named_modules():
        if name.startswith('decoder'):
            module.apply(init_decoder_weights)
    
    # 2. 特别处理mask_token和decoder_pos_embed
    if hasattr(model, 'mask_token'):
        nn.init.normal_(model.mask_token, std=0.02)
    
    if hasattr(model, 'decoder_pos_embed'):
        nn.init.normal_(model.decoder_pos_embed, std=0.02)
    
    # 3. 最后的预测层使用更保守的初始化
    if hasattr(model, 'decoder_pred'):
        nn.init.normal_(model.decoder_pred.weight, std=0.01)
        nn.init.constant_(model.decoder_pred.bias, 0)
    
    model = model.to(device)
    model.eval()
    
    print("✅ 改进模型创建完成")
    return model, device

def create_simple_decoder_model():
    """创建简化的解码器模型"""
    print("\n🎯 创建简化解码器模型...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 使用默认参数创建模型，然后手动修改解码器
    model = models_mae.mae_vit_base_patch16()
    
    # 加载编码器权重
    pretrain_path = 'pretrained_models/mae_pretrain_vit_base.pth'
    if os.path.exists(pretrain_path):
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        
        encoder_state_dict = {}
        for key, value in checkpoint['model'].items():
            if not key.startswith('decoder') and key != 'mask_token':
                encoder_state_dict[key] = value
        
        model.load_state_dict(encoder_state_dict, strict=False)
        print("✅ 编码器权重加载成功")
    
    # 简化的解码器初始化
    def simple_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)  # 更小的gain
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    for name, module in model.named_modules():
        if name.startswith('decoder'):
            module.apply(simple_init)
    
    model = model.to(device)
    model.eval()
    
    print("✅ 简化模型创建完成")
    return model, device

def test_improved_reconstruction():
    """测试改进的重建效果"""
    print("\n🧪 测试改进的重建效果...")
    
    # 创建不同的模型
    models = {}
    
    # 1. 原始模型（随机解码器）
    print("📦 加载原始模型...")
    original_model = models_mae.mae_vit_base_patch16()
    pretrain_path = 'pretrained_models/mae_pretrain_vit_base.pth'
    if os.path.exists(pretrain_path):
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        encoder_state_dict = {}
        for key, value in checkpoint['model'].items():
            if not key.startswith('decoder') and key != 'mask_token':
                encoder_state_dict[key] = value
        original_model.load_state_dict(encoder_state_dict, strict=False)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    original_model = original_model.to(device)
    original_model.eval()
    models['原始模型'] = original_model
    
    # 2. 改进初始化模型
    improved_model, _ = create_improved_mae_model()
    models['改进初始化'] = improved_model
    
    # 3. 简化解码器模型
    simple_model, _ = create_simple_decoder_model()
    models['简化解码器'] = simple_model
    
    # 创建测试图像
    test_img = create_clean_test_image(device)
    
    # 反归一化
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # 测试重建
    fig, axes = plt.subplots(len(models), 4, figsize=(16, len(models)*4))
    
    if len(models) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (model_name, model) in enumerate(models.items()):
        print(f"\n  🔍 测试 {model_name}...")
        
        with torch.no_grad():
            # 原始图像
            original_display = torch.clamp(inv_normalize(test_img[0]).cpu(), 0, 1)
            axes[i, 0].imshow(original_display.permute(1, 2, 0))
            axes[i, 0].set_title(f'{model_name}\n原始图像')
            axes[i, 0].axis('off')
            
            # MAE重建
            loss, pred, mask = model(test_img, mask_ratio=0.75)
            reconstructed = model.unpatchify(pred)
            
            # 创建掩码图像
            mask_vis = mask.detach()
            mask_vis = mask_vis.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
            mask_vis = model.unpatchify(mask_vis)
            
            masked_img = original_display * (1 - mask_vis[0].cpu()) + mask_vis[0].cpu() * 0.5
            
            axes[i, 1].imshow(masked_img.permute(1, 2, 0))
            axes[i, 1].set_title('掩码图像 (75%)')
            axes[i, 1].axis('off')
            
            # 重建图像
            reconstructed_display = torch.clamp(inv_normalize(reconstructed[0]).cpu(), 0, 1)
            axes[i, 2].imshow(reconstructed_display.permute(1, 2, 0))
            axes[i, 2].set_title(f'重建图像\n损失:{loss.item():.3f}')
            axes[i, 2].axis('off')
            
            # 重建误差
            error = torch.abs(original_display - reconstructed_display)
            error_display = error.mean(dim=0)
            im = axes[i, 3].imshow(error_display, cmap='hot')
            axes[i, 3].set_title(f'重建误差\n均值:{error.mean():.3f}')
            axes[i, 3].axis('off')
            plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
            
            print(f"    损失: {loss.item():.4f}")
            print(f"    预测范围: [{pred.min():.3f}, {pred.max():.3f}]")
            print(f"    重建误差: {error.mean():.4f}")
    
    plt.tight_layout()
    
    # 保存结果
    output_path = 'improved_mae_reconstruction.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 改进结果保存: {output_path}")
    
    try:
        plt.show()
    except:
        print("💡 如果要查看图像，请在支持图形界面的环境中运行")
    
    return output_path

def create_clean_test_image(device):
    """创建清晰的测试图像"""
    img = torch.zeros(1, 3, 224, 224, device=device)
    
    # 创建更清晰的图案
    # 1. 背景渐变
    for i in range(224):
        for j in range(224):
            r = (0.3 - 0.485) / 0.229
            g = (0.4 - 0.456) / 0.224
            b = (0.5 - 0.406) / 0.225
            img[0, 0, i, j] = r
            img[0, 1, i, j] = g
            img[0, 2, i, j] = b
    
    # 2. 添加清晰的几何图案
    # 圆形
    y, x = torch.meshgrid(torch.arange(224, device=device), torch.arange(224, device=device), indexing='ij')
    circle_mask = (x - 112)**2 + (y - 112)**2 <= 50**2
    
    img[0, 0][circle_mask] = (0.8 - 0.485) / 0.229  # 红色
    img[0, 1][circle_mask] = (0.2 - 0.456) / 0.224
    img[0, 2][circle_mask] = (0.2 - 0.406) / 0.225
    
    # 矩形
    rect_mask = (x >= 50) & (x <= 100) & (y >= 50) & (y <= 100)
    img[0, 0][rect_mask] = (0.2 - 0.485) / 0.229
    img[0, 1][rect_mask] = (0.8 - 0.456) / 0.224  # 绿色
    img[0, 2][rect_mask] = (0.2 - 0.406) / 0.225
    
    return img

def explain_noise_problem():
    """解释噪声问题"""
    print("\n📚 噪声问题详细解释:")
    print("=" * 60)
    
    print("🔍 问题根源:")
    print("1. Facebook官方预训练模型只包含编码器权重")
    print("2. 解码器使用随机初始化，没有学会如何重建像素")
    print("3. 编码器输出的特征是抽象的，解码器不知道如何解释")
    print()
    
    print("🎯 为什么会产生噪声:")
    print("• 随机初始化的解码器权重导致输出不稳定")
    print("• 解码器没有学会从特征到像素的映射关系")
    print("• 预测值的范围和分布不合理")
    print()
    
    print("🛠️ 改进策略:")
    print("1. 更好的权重初始化 - 使用更小的标准差")
    print("2. 简化解码器架构 - 减少参数数量")
    print("3. 渐进式训练 - 先训练简单任务再复杂任务")
    print("4. 使用完整预训练模型 - 包含解码器权重")
    print()
    
    print("💡 实际解决方案:")
    print("• 下载完整的MAE预训练模型（如果存在）")
    print("• 在你的数据上微调解码器")
    print("• 使用其他预训练的图像重建模型")
    print("• 考虑使用ViT进行特征提取而非重建")

def main():
    """主函数"""
    print("🛠️ 改进的MAE重建演示")
    print("=" * 50)
    
    # 设置环境
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 测试改进的重建
    output_path = test_improved_reconstruction()
    
    # 解释问题
    explain_noise_problem()
    
    print(f"\n🎯 总结:")
    print("✅ 通过改进初始化，噪声可能会减少")
    print("❌ 但根本问题是解码器没有预训练")
    print("💡 最好的解决方案是使用完整预训练模型")
    print(f"📁 改进结果: {output_path}")

if __name__ == "__main__":
    main()
