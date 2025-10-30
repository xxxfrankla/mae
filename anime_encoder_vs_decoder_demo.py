#!/usr/bin/env python3
"""
使用Anime Diffusion数据集演示MAE编码器vs解码器
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_anime_dataset_sample():
    """从anime diffusion数据集加载样本"""
    print("🎨 从Anime Diffusion数据集加载样本...")
    
    # 检查数据集是否存在
    dataset_paths = [
        '/Users/tdu/Documents/GitHub/mae/test_dataset',
        './test_dataset'
    ]
    
    dataset_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None:
        print("❌ 未找到anime数据集，创建示例图像...")
        return create_anime_style_examples()
    
    # 获取图像文件
    image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) == 0:
        print("❌ 数据集中没有图像文件，创建示例图像...")
        return create_anime_style_examples()
    
    # 随机选择几张图像
    selected_files = np.random.choice(image_files, min(3, len(image_files)), replace=False)
    
    images = []
    for filename in selected_files:
        try:
            img_path = os.path.join(dataset_path, filename)
            img = Image.open(img_path).convert('RGB')
            # 调整大小到224x224
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            images.append((f"Anime_{filename[:10]}", np.array(img) / 255.0))
            print(f"  ✅ 加载: {filename}")
        except Exception as e:
            print(f"  ❌ 加载失败 {filename}: {e}")
    
    return images

def create_anime_style_examples():
    """创建动漫风格的示例图像"""
    print("  创建动漫风格示例图像...")
    
    images = []
    
    # 示例1：简单的动漫脸部轮廓
    img1 = np.ones((224, 224, 3)) * 0.95  # 浅色背景
    
    # 脸部轮廓（椭圆）
    y, x = np.ogrid[:224, :224]
    center_y, center_x = 112, 112
    face_mask = ((x - center_x)/60)**2 + ((y - center_y)/80)**2 <= 1
    img1[face_mask] = [1.0, 0.9, 0.8]  # 肤色
    
    # 眼睛
    eye1_mask = ((x - 90)/8)**2 + ((y - 90)/12)**2 <= 1
    eye2_mask = ((x - 134)/8)**2 + ((y - 90)/12)**2 <= 1
    img1[eye1_mask] = [0.1, 0.1, 0.1]  # 黑色眼睛
    img1[eye2_mask] = [0.1, 0.1, 0.1]
    
    # 嘴巴
    mouth_mask = ((x - 112)/15)**2 + ((y - 140)/5)**2 <= 1
    img1[mouth_mask] = [0.8, 0.3, 0.3]  # 红色嘴巴
    
    images.append(("动漫脸部", img1))
    
    # 示例2：彩色几何图案
    img2 = np.zeros((224, 224, 3))
    for i in range(0, 224, 28):
        for j in range(0, 224, 28):
            color = [(i/224), (j/224), 0.8]
            img2[i:i+28, j:j+28] = color
    images.append(("彩色方格", img2))
    
    # 示例3：星空背景
    img3 = np.zeros((224, 224, 3))
    img3[:, :, 2] = 0.2  # 深蓝背景
    
    # 随机星星
    np.random.seed(42)
    for _ in range(50):
        x_star = np.random.randint(0, 224)
        y_star = np.random.randint(0, 224)
        size = np.random.randint(1, 4)
        brightness = np.random.uniform(0.5, 1.0)
        img3[max(0, y_star-size):min(224, y_star+size), 
             max(0, x_star-size):min(224, x_star+size)] = [brightness, brightness, brightness]
    
    images.append(("星空背景", img3))
    
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

def demonstrate_encoder_vs_decoder(model, device, images):
    """演示编码器 vs 完整MAE的区别"""
    print("\n🔍 演示：编码器能否重建图像？")
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 反归一化
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    num_images = len(images)
    fig, axes = plt.subplots(4, num_images, figsize=(num_images*4, 16))
    
    if num_images == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (name, img) in enumerate(images):
        print(f"\n  处理图像: {name}")
        
        # 转换为tensor
        img_tensor = transform(img.astype(np.float32)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 1. 显示原始图像
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'{name}\n原始图像')
            axes[0, i].axis('off')
            
            # 2. 只用编码器提取特征
            print("    🔄 编码器特征提取...")
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
            
            # 尝试直接从编码器特征"重建"（这是不可能的！）
            # 我们只能可视化特征
            patch_features = encoded_features[:, 1:, :]  # 去掉cls token
            feature_map = patch_features.mean(dim=2).cpu().numpy().reshape(14, 14)
            
            im1 = axes[1, i].imshow(feature_map, cmap='viridis')
            axes[1, i].set_title('编码器特征\n(无法重建图像!)')
            axes[1, i].axis('off')
            plt.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)
            
            # 3. 模拟完整MAE的掩码过程
            print("    🎭 模拟掩码过程...")
            mask_ratio = 0.75
            
            # 创建随机掩码
            N, L, D = encoded_features.shape
            len_keep = int(L * (1 - mask_ratio))
            
            noise = torch.rand(N, L, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            
            # 创建掩码可视化
            mask = torch.ones([N, L], device=device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)
            
            # 可视化掩码
            mask_vis = mask[:, 1:].cpu().numpy().reshape(14, 14)  # 去掉cls token
            
            axes[2, i].imshow(mask_vis, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[2, i].set_title(f'掩码模式\n({mask_ratio*100:.0f}% 被掩盖)')
            axes[2, i].axis('off')
            
            # 4. 说明需要解码器
            axes[3, i].text(0.5, 0.7, '❌ 只有编码器', ha='center', va='center', 
                           transform=axes[3, i].transAxes, fontsize=14, color='red')
            axes[3, i].text(0.5, 0.5, '无法重建图像!', ha='center', va='center',
                           transform=axes[3, i].transAxes, fontsize=12)
            axes[3, i].text(0.5, 0.3, '✅ 需要解码器', ha='center', va='center',
                           transform=axes[3, i].transAxes, fontsize=14, color='green')
            axes[3, i].text(0.5, 0.1, '才能重建像素', ha='center', va='center',
                           transform=axes[3, i].transAxes, fontsize=12)
            axes[3, i].set_xlim(0, 1)
            axes[3, i].set_ylim(0, 1)
            axes[3, i].axis('off')
            axes[3, i].set_title('重建需要解码器!')
            
            print(f"    编码器特征维度: {encoded_features.shape}")
            print(f"    特征均值: {encoded_features.mean().item():.4f}")
    
    plt.tight_layout()
    
    # 保存结果
    output_path = 'anime_encoder_vs_decoder_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 演示结果已保存: {output_path}")
    
    try:
        plt.show()
    except:
        print("💡 如果要查看图像，请在支持图形界面的环境中运行")
    
    return output_path

def explain_mae_architecture():
    """解释MAE架构"""
    print("\n📚 MAE架构解释:")
    print("=" * 50)
    print("🔍 编码器 (Encoder):")
    print("  • 输入: 原始图像的可见patches")
    print("  • 功能: 提取高级语义特征")
    print("  • 输出: 抽象的特征表示")
    print("  • ❌ 无法直接重建像素!")
    print()
    print("🎨 解码器 (Decoder):")
    print("  • 输入: 编码器特征 + mask tokens")
    print("  • 功能: 从特征重建像素")
    print("  • 输出: 重建的图像patches")
    print("  • ✅ 负责像素级重建!")
    print()
    print("🎭 完整MAE流程:")
    print("  1. 图像分patch → 随机掩码75%")
    print("  2. 可见patches → 编码器 → 特征")
    print("  3. 特征 + mask tokens → 解码器 → 重建")
    print("  4. 计算重建损失，训练模型")

def main():
    """主函数"""
    print("🎭 Anime Diffusion数据集 - MAE编码器vs解码器演示")
    print("=" * 60)
    
    # 设置环境
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 加载anime数据集样本
    anime_images = load_anime_dataset_sample()
    
    # 加载MAE模型
    model, device = load_mae_model()
    
    # 演示编码器vs解码器
    output_path = demonstrate_encoder_vs_decoder(model, device, anime_images)
    
    # 解释架构
    explain_mae_architecture()
    
    print(f"\n🎉 演示完成!")
    print(f"📁 结果图像: {output_path}")
    print("\n💡 关键结论:")
    print("❌ 只用编码器无法重建图像")
    print("✅ 编码器只能提取抽象特征")
    print("🎨 解码器负责从特征重建像素")
    print("🔄 完整的MAE = 编码器 + 解码器")

if __name__ == "__main__":
    main()

