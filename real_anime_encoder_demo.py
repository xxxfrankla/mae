#!/usr/bin/env python3
"""
使用真实Anime数据集演示MAE编码器vs解码器
回答问题：只用编码器能否重建图像？
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import models_mae
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_real_anime_samples():
    """从真实anime数据集加载样本"""
    print("🎨 从真实Anime数据集加载样本...")
    
    dataset_path = '/Users/tdu/Documents/GitHub/mae/test_dataset/train'
    
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return []
    
    # 获取所有类别
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    classes.sort()
    
    print(f"  发现 {len(classes)} 个类别: {classes}")
    
    images = []
    
    # 从每个类别随机选择一张图像
    for class_name in classes[:3]:  # 只取前3个类别
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith('.png')]
        
        if len(image_files) > 0:
            # 随机选择一张图像
            selected_file = random.choice(image_files)
            img_path = os.path.join(class_path, selected_file)
            
            try:
                img = Image.open(img_path).convert('RGB')
                # 调整大小到224x224
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                images.append((f"Anime_{class_name}", np.array(img) / 255.0))
                print(f"  ✅ 加载: {class_name}/{selected_file}")
            except Exception as e:
                print(f"  ❌ 加载失败 {img_path}: {e}")
    
    return images

def load_mae_model():
    """加载MAE模型"""
    print("\n🤖 加载MAE模型...")
    
    # 检查设备
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✅ 使用 Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        print(f"✅ 使用 CPU")
    
    # 创建模型
    model = models_mae.mae_vit_base_patch16()
    
    # 加载预训练权重（只有编码器部分）
    pretrain_path = 'pretrained_models/mae_pretrain_vit_base.pth'
    if os.path.exists(pretrain_path):
        print(f"📥 加载预训练权重: {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        
        # 只加载编码器权重
        encoder_state_dict = {}
        for key, value in checkpoint['model'].items():
            if not key.startswith('decoder') and key != 'mask_token':
                encoder_state_dict[key] = value
        
        model.load_state_dict(encoder_state_dict, strict=False)
        print("✅ 编码器权重加载成功")
    else:
        print("⚠️  使用随机初始化的权重")
    
    model = model.to(device)
    model.eval()
    
    return model, device

def demonstrate_why_encoder_cannot_reconstruct(model, device, images):
    """详细演示为什么编码器无法重建图像"""
    print("\n🔍 详细分析：为什么编码器无法重建图像？")
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    num_images = len(images)
    fig, axes = plt.subplots(5, num_images, figsize=(num_images*4, 20))
    
    if num_images == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (name, img) in enumerate(images):
        print(f"\n  分析图像: {name}")
        
        # 转换为tensor
        img_tensor = transform(img.astype(np.float32)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 1. 原始图像
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'{name}\n原始图像 (224×224×3)')
            axes[0, i].axis('off')
            
            # 2. 图像分块可视化
            print("    📦 图像分块过程...")
            patches = model.patchify(img_tensor)  # (N, L, patch_size**2 * 3)
            print(f"      分块后形状: {patches.shape}")  # (1, 196, 768)
            
            # 可视化前16个patches
            patch_grid = np.zeros((4*16, 4*16, 3))
            for p in range(16):
                patch_data = patches[0, p].cpu().numpy().reshape(16, 16, 3)
                # 反归一化显示
                patch_data = patch_data * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                patch_data = np.clip(patch_data, 0, 1)
                
                row, col = p // 4, p % 4
                patch_grid[row*16:(row+1)*16, col*16:(col+1)*16] = patch_data
            
            axes[1, i].imshow(patch_grid)
            axes[1, i].set_title('前16个Patches\n(16×16像素块)')
            axes[1, i].axis('off')
            
            # 3. 编码器特征提取
            print("    🧠 编码器特征提取...")
            x = model.patch_embed(img_tensor)  # (N, L, D)
            x = x + model.pos_embed[:, 1:, :]  # 添加位置编码
            
            # 添加cls token
            cls_token = model.cls_token + model.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            # 通过编码器层
            for blk in model.blocks:
                x = blk(x)
            encoded_features = model.norm(x)
            
            print(f"      编码器输出形状: {encoded_features.shape}")  # (1, 197, 768)
            
            # 可视化编码器特征
            patch_features = encoded_features[:, 1:, :].cpu().numpy()  # 去掉cls token
            feature_mean = patch_features[0].mean(axis=1).reshape(14, 14)
            
            im1 = axes[2, i].imshow(feature_mean, cmap='viridis')
            axes[2, i].set_title('编码器特征\n(抽象语义表示)')
            axes[2, i].axis('off')
            plt.colorbar(im1, ax=axes[2, i], fraction=0.046, pad=0.04)
            
            # 4. 尝试"重建"的问题
            print("    ❌ 尝试从特征重建...")
            
            # 编码器特征是768维的抽象表示
            # 而原始patch是16×16×3=768维的像素值
            # 虽然维度相同，但语义完全不同！
            
            # 错误的"重建"尝试：直接将特征当作像素
            fake_reconstruction = patch_features[0].reshape(14, 14, 768)[:, :, :3]
            fake_reconstruction = (fake_reconstruction - fake_reconstruction.min()) / (fake_reconstruction.max() - fake_reconstruction.min())
            
            axes[3, i].imshow(fake_reconstruction)
            axes[3, i].set_title('错误的"重建"\n(特征≠像素!)')
            axes[3, i].axis('off')
            
            # 5. 解释为什么需要解码器
            axes[4, i].text(0.5, 0.9, '🧠 编码器输出:', ha='center', va='center',
                           transform=axes[4, i].transAxes, fontsize=12, weight='bold')
            axes[4, i].text(0.5, 0.8, '768维抽象特征', ha='center', va='center',
                           transform=axes[4, i].transAxes, fontsize=11)
            axes[4, i].text(0.5, 0.7, '(语义信息)', ha='center', va='center',
                           transform=axes[4, i].transAxes, fontsize=10, style='italic')
            
            axes[4, i].text(0.5, 0.5, '≠', ha='center', va='center',
                           transform=axes[4, i].transAxes, fontsize=20, color='red')
            
            axes[4, i].text(0.5, 0.3, '🎨 需要的输出:', ha='center', va='center',
                           transform=axes[4, i].transAxes, fontsize=12, weight='bold')
            axes[4, i].text(0.5, 0.2, '768维像素值', ha='center', va='center',
                           transform=axes[4, i].transAxes, fontsize=11)
            axes[4, i].text(0.5, 0.1, '(16×16×3 RGB)', ha='center', va='center',
                           transform=axes[4, i].transAxes, fontsize=10, style='italic')
            
            axes[4, i].set_xlim(0, 1)
            axes[4, i].set_ylim(0, 1)
            axes[4, i].axis('off')
            axes[4, i].set_title('需要解码器转换!')
            
            print(f"      特征统计: 均值={encoded_features.mean().item():.4f}, 标准差={encoded_features.std().item():.4f}")
    
    plt.tight_layout()
    
    # 保存结果
    output_path = 'real_anime_encoder_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 分析结果已保存: {output_path}")
    
    try:
        plt.show()
    except:
        print("💡 如果要查看图像，请在支持图形界面的环境中运行")
    
    return output_path

def explain_mae_detailed():
    """详细解释MAE工作原理"""
    print("\n📚 MAE详细工作原理:")
    print("=" * 60)
    
    print("🔍 1. 编码器 (Encoder) - 特征提取器:")
    print("   输入: 可见的图像patches (25%)")
    print("   处理: 多层Transformer → 抽象语义特征")
    print("   输出: 768维特征向量 (每个patch)")
    print("   作用: 理解图像内容，但不能生成像素")
    print()
    
    print("🎨 2. 解码器 (Decoder) - 像素生成器:")
    print("   输入: 编码器特征 + mask tokens")
    print("   处理: 轻量级Transformer → 像素预测")
    print("   输出: 768维像素值 (16×16×3)")
    print("   作用: 从语义特征重建具体像素")
    print()
    
    print("🎭 3. 为什么编码器无法重建:")
    print("   • 编码器学习的是抽象语义特征")
    print("   • 特征表示物体、纹理、关系等高级概念")
    print("   • 像素是具体的颜色值 (0-255)")
    print("   • 语义特征 ≠ 像素值 (虽然维度可能相同)")
    print()
    
    print("🔄 4. 完整MAE训练过程:")
    print("   Step 1: 随机掩码75%的patches")
    print("   Step 2: 编码器处理可见patches → 特征")
    print("   Step 3: 解码器接收特征+mask tokens → 重建")
    print("   Step 4: 计算重建损失，反向传播训练")
    print()
    
    print("💡 5. 关键洞察:")
    print("   • 编码器专注于理解 (understanding)")
    print("   • 解码器专注于生成 (generation)")
    print("   • 两者分工合作，缺一不可")

def main():
    """主函数"""
    print("🎭 真实Anime数据集 - MAE编码器深度分析")
    print("=" * 60)
    
    # 设置环境
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    random.seed(42)  # 确保可重现
    
    # 加载真实anime数据集样本
    anime_images = load_real_anime_samples()
    
    if len(anime_images) == 0:
        print("❌ 没有加载到图像，请检查数据集路径")
        return
    
    # 加载MAE模型
    model, device = load_mae_model()
    
    # 详细演示分析
    output_path = demonstrate_why_encoder_cannot_reconstruct(model, device, anime_images)
    
    # 详细解释
    explain_mae_detailed()
    
    print(f"\n🎉 分析完成!")
    print(f"📁 结果图像: {output_path}")
    print("\n🎯 核心结论:")
    print("❌ 编码器输出抽象语义特征，不是像素值")
    print("🎨 解码器负责从特征生成具体像素")
    print("🔄 MAE = 编码器(理解) + 解码器(生成)")
    print("💡 这就是为什么只有编码器无法重建图像!")

if __name__ == "__main__":
    main()

