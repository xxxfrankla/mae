#!/usr/bin/env python3
"""
使用AnimeDiffusion数据集的最完整MAE演示
展示编码器vs解码器，回答：只用编码器能否重建图像？
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import models_mae
from animediffusion_dataset_loader import create_animediffusion_dataloader
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
        decoder_missing_keys = []
        
        for key, value in checkpoint['model'].items():
            if not key.startswith('decoder') and key != 'mask_token':
                encoder_state_dict[key] = value
            else:
                decoder_missing_keys.append(key)
        
        model.load_state_dict(encoder_state_dict, strict=False)
        print("✅ 编码器权重加载成功")
        print(f"⚠️  解码器权重缺失: {len(decoder_missing_keys)} 个参数")
    else:
        print("⚠️  使用随机初始化的权重")
    
    model = model.to(device)
    model.eval()
    
    return model, device

def load_animediffusion_samples():
    """使用AnimeDiffusion数据集加载器获取样本"""
    print("\n🎌 使用AnimeDiffusion数据集加载器...")
    
    try:
        # 创建数据加载器，只加载少量样本用于演示
        dataloader, dataset = create_animediffusion_dataloader(
            batch_size=6,  # 一次加载6张图片
            max_samples=50,  # 只从前50张中选择
            input_size=224,
            num_workers=0  # 避免多进程问题
        )
        
        if dataloader is None:
            print("❌ 数据加载器创建失败")
            return None
        
        # 获取第一个批次
        for images, _ in dataloader:
            print(f"✅ 成功加载 {images.shape[0]} 张AnimeDiffusion图片")
            print(f"   图像形状: {images.shape}")
            print(f"   数据范围: [{images.min():.3f}, {images.max():.3f}]")
            return images[:3]  # 只取前3张用于演示
            
    except Exception as e:
        print(f"❌ AnimeDiffusion数据集加载失败: {e}")
        print("   可能需要网络连接或HuggingFace账户")
        return None

def create_fallback_anime_samples():
    """创建备用的动漫风格样本"""
    print("🎨 创建备用动漫风格样本...")
    
    # 创建标准化的tensor
    samples = []
    
    # 样本1：动漫人物脸部
    img1 = torch.zeros(3, 224, 224)
    
    # 脸部轮廓
    y, x = torch.meshgrid(torch.arange(224), torch.arange(224), indexing='ij')
    face_mask = ((x - 112)/60)**2 + ((y - 112)/80)**2 <= 1
    
    # 肤色 (归一化后的值)
    img1[0][face_mask] = (1.0 - 0.485) / 0.229  # R
    img1[1][face_mask] = (0.9 - 0.456) / 0.224  # G  
    img1[2][face_mask] = (0.8 - 0.406) / 0.225  # B
    
    # 眼睛
    eye1_mask = ((x - 90)/8)**2 + ((y - 90)/12)**2 <= 1
    eye2_mask = ((x - 134)/8)**2 + ((y - 90)/12)**2 <= 1
    for c in range(3):
        img1[c][eye1_mask] = (0.1 - [0.485, 0.456, 0.406][c]) / [0.229, 0.224, 0.225][c]
        img1[c][eye2_mask] = (0.1 - [0.485, 0.456, 0.406][c]) / [0.229, 0.224, 0.225][c]
    
    samples.append(img1.unsqueeze(0))
    
    # 样本2：彩色渐变
    img2 = torch.zeros(3, 224, 224)
    for i in range(224):
        for j in range(224):
            r = (i / 224 - 0.485) / 0.229
            g = (j / 224 - 0.456) / 0.224  
            b = (0.8 - 0.406) / 0.225
            img2[0, i, j] = r
            img2[1, i, j] = g
            img2[2, i, j] = b
    
    samples.append(img2.unsqueeze(0))
    
    # 样本3：几何图案
    img3 = torch.zeros(3, 224, 224)
    for i in range(0, 224, 28):
        for j in range(0, 224, 28):
            color_r = (i/224 - 0.485) / 0.229
            color_g = (j/224 - 0.456) / 0.224
            color_b = (0.7 - 0.406) / 0.225
            img3[0, i:i+28, j:j+28] = color_r
            img3[1, i:i+28, j:j+28] = color_g  
            img3[2, i:i+28, j:j+28] = color_b
    
    samples.append(img3.unsqueeze(0))
    
    # 合并所有样本
    return torch.cat(samples, dim=0)

def demonstrate_complete_mae_analysis(model, device, images):
    """完整的MAE分析演示"""
    print("\n🔍 完整MAE分析：编码器 vs 解码器")
    
    # 反归一化用于显示
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    num_images = min(3, images.shape[0])
    fig, axes = plt.subplots(6, num_images, figsize=(num_images*4, 24))
    
    if num_images == 1:
        axes = axes.reshape(-1, 1)
    
    images = images[:num_images].to(device)
    
    for i in range(num_images):
        img_tensor = images[i:i+1]
        
        print(f"\n  分析图像 {i+1}...")
        
        with torch.no_grad():
            # 1. 原始图像显示
            original_display = torch.clamp(inv_normalize(img_tensor[0]).cpu(), 0, 1)
            axes[0, i].imshow(original_display.permute(1, 2, 0))
            axes[0, i].set_title(f'AnimeDiffusion图像 {i+1}\n原始高质量动漫图片')
            axes[0, i].axis('off')
            
            # 2. 图像分块过程
            print("    📦 图像分块...")
            patches = model.patchify(img_tensor)  # (1, 196, 768)
            
            # 可视化前16个patches
            patch_display = torch.zeros(4*16, 4*16, 3)
            for p in range(16):
                if p < patches.shape[1]:
                    patch_data = patches[0, p].cpu().reshape(16, 16, 3)
                    # 反归一化
                    patch_data = patch_data * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                    patch_data = torch.clamp(patch_data, 0, 1)
                    
                    row, col = p // 4, p % 4
                    patch_display[row*16:(row+1)*16, col*16:(col+1)*16] = patch_data
            
            axes[1, i].imshow(patch_display)
            axes[1, i].set_title(f'图像分块\n前16个16×16 patches')
            axes[1, i].axis('off')
            
            # 3. 编码器特征提取
            print("    🧠 编码器特征提取...")
            x = model.patch_embed(img_tensor)
            x = x + model.pos_embed[:, 1:, :]
            
            # 添加cls token
            cls_token = model.cls_token + model.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            # 通过编码器
            for blk in model.blocks:
                x = blk(x)
            encoded_features = model.norm(x)
            
            # 可视化编码器特征
            patch_features = encoded_features[:, 1:, :].cpu().numpy()
            feature_map = patch_features[0].mean(axis=1).reshape(14, 14)
            
            im1 = axes[2, i].imshow(feature_map, cmap='viridis')
            axes[2, i].set_title('编码器输出特征\n768维抽象语义表示')
            axes[2, i].axis('off')
            plt.colorbar(im1, ax=axes[2, i], fraction=0.046, pad=0.04)
            
            # 4. 掩码模拟
            print("    🎭 模拟MAE掩码过程...")
            mask_ratio = 0.75
            N, L, D = encoded_features.shape
            len_keep = int(L * (1 - mask_ratio))
            
            # 创建随机掩码
            noise = torch.rand(N, L-1, device=device)  # 不包括cls token
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            
            # 创建掩码
            mask = torch.ones([N, L-1], device=device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)
            
            # 可视化掩码
            mask_vis = mask[0].cpu().numpy().reshape(14, 14)
            
            axes[3, i].imshow(mask_vis, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[3, i].set_title(f'MAE掩码模式\n{mask_ratio*100:.0f}% patches被掩盖')
            axes[3, i].axis('off')
            
            # 5. 编码器限制说明
            axes[4, i].text(0.5, 0.8, '❌ 编码器无法重建', ha='center', va='center',
                           transform=axes[4, i].transAxes, fontsize=14, color='red', weight='bold')
            axes[4, i].text(0.5, 0.65, '编码器输出:', ha='center', va='center',
                           transform=axes[4, i].transAxes, fontsize=12)
            axes[4, i].text(0.5, 0.55, '768维语义特征', ha='center', va='center',
                           transform=axes[4, i].transAxes, fontsize=11)
            axes[4, i].text(0.5, 0.4, '≠', ha='center', va='center',
                           transform=axes[4, i].transAxes, fontsize=20, color='red')
            axes[4, i].text(0.5, 0.25, '需要的输出:', ha='center', va='center',
                           transform=axes[4, i].transAxes, fontsize=12)
            axes[4, i].text(0.5, 0.15, '768维像素值', ha='center', va='center',
                           transform=axes[4, i].transAxes, fontsize=11)
            axes[4, i].set_xlim(0, 1)
            axes[4, i].set_ylim(0, 1)
            axes[4, i].axis('off')
            axes[4, i].set_title('编码器的限制')
            
            # 6. 完整MAE需要解码器
            axes[5, i].text(0.5, 0.9, '✅ 完整MAE架构', ha='center', va='center',
                           transform=axes[5, i].transAxes, fontsize=14, color='green', weight='bold')
            axes[5, i].text(0.5, 0.75, '编码器: 理解图像', ha='center', va='center',
                           transform=axes[5, i].transAxes, fontsize=11)
            axes[5, i].text(0.5, 0.65, '↓', ha='center', va='center',
                           transform=axes[5, i].transAxes, fontsize=16)
            axes[5, i].text(0.5, 0.55, '抽象特征', ha='center', va='center',
                           transform=axes[5, i].transAxes, fontsize=11)
            axes[5, i].text(0.5, 0.45, '↓', ha='center', va='center',
                           transform=axes[5, i].transAxes, fontsize=16)
            axes[5, i].text(0.5, 0.35, '解码器: 生成像素', ha='center', va='center',
                           transform=axes[5, i].transAxes, fontsize=11)
            axes[5, i].text(0.5, 0.25, '↓', ha='center', va='center',
                           transform=axes[5, i].transAxes, fontsize=16)
            axes[5, i].text(0.5, 0.15, '重建图像', ha='center', va='center',
                           transform=axes[5, i].transAxes, fontsize=11, color='green')
            axes[5, i].set_xlim(0, 1)
            axes[5, i].set_ylim(0, 1)
            axes[5, i].axis('off')
            axes[5, i].set_title('需要完整架构')
            
            # 打印统计信息
            print(f"      编码器特征统计: 均值={encoded_features.mean().item():.4f}, 标准差={encoded_features.std().item():.4f}")
            print(f"      特征维度: {encoded_features.shape}")
    
    plt.tight_layout()
    
    # 保存结果
    output_path = 'complete_animediffusion_mae_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 完整分析结果已保存: {output_path}")
    
    try:
        plt.show()
    except:
        print("💡 如果要查看图像，请在支持图形界面的环境中运行")
    
    return output_path

def explain_complete_mae():
    """完整解释MAE原理"""
    print("\n📚 MAE完整原理解析:")
    print("=" * 60)
    
    print("🎌 AnimeDiffusion数据集特点:")
    print("   • 高质量动漫图片 (1920×1080)")
    print("   • 丰富的视觉细节和色彩")
    print("   • 复杂的纹理和结构")
    print("   • 适合测试重建能力")
    print()
    
    print("🔍 MAE编码器分析:")
    print("   输入: 25%可见patches (49/196个)")
    print("   处理: 12层Transformer编码器")
    print("   输出: 768维抽象特征向量")
    print("   功能: 理解图像语义内容")
    print("   ❌ 无法直接生成像素!")
    print()
    
    print("🎨 MAE解码器作用:")
    print("   输入: 编码器特征 + 147个mask tokens")
    print("   处理: 8层轻量级Transformer")
    print("   输出: 196个768维像素预测")
    print("   功能: 从特征重建具体像素")
    print("   ✅ 负责像素级重建!")
    print()
    
    print("🎯 关键技术洞察:")
    print("   1. 编码器学习语义表示 (what)")
    print("   2. 解码器学习像素生成 (how)")
    print("   3. 75%掩码率强迫模型理解全局结构")
    print("   4. 预训练学到的特征可用于下游任务")
    print()
    
    print("💡 实际应用价值:")
    print("   • 编码器: 特征提取、分类、检索")
    print("   • 完整MAE: 图像修复、去噪、编辑")
    print("   • 预训练权重: 提升下游任务性能")

def main():
    """主函数"""
    print("🎌 AnimeDiffusion数据集 - 最完整MAE演示")
    print("=" * 60)
    
    # 设置环境
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    random.seed(42)
    
    # 加载MAE模型
    model, device = load_mae_model()
    
    # 尝试加载AnimeDiffusion数据集
    print("\n🎯 尝试加载真实AnimeDiffusion数据集...")
    images = load_animediffusion_samples()
    
    if images is None:
        print("\n🎨 使用备用动漫风格样本...")
        images = create_fallback_anime_samples()
    
    # 完整演示分析
    output_path = demonstrate_complete_mae_analysis(model, device, images)
    
    # 完整原理解释
    explain_complete_mae()
    
    print(f"\n🎉 最完整演示完成!")
    print(f"📁 结果图像: {output_path}")
    print("\n🎯 最终结论:")
    print("❌ 编码器只能提取抽象语义特征")
    print("🎨 解码器才能从特征生成像素")
    print("🔄 完整MAE = 编码器(理解) + 解码器(重建)")
    print("💡 这就是为什么只有编码器无法重建AnimeDiffusion图像!")

if __name__ == "__main__":
    main()

