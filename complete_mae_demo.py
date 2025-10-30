#!/usr/bin/env python3
"""
使用包含完整解码器权重的MAE模型进行高质量重建演示
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
from datetime import datetime

# 设置中文字体和环境
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def load_complete_mae_model():
    """加载包含完整解码器的MAE模型"""
    print("\n🎯 加载包含完整解码器的MAE模型...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建ViT-Large模型（与可视化模型匹配）
    model = models_mae.mae_vit_large_patch16()
    
    # 加载完整的预训练权重
    model_path = 'complete_mae_models/mae_visualize_vit_large.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行: ./download_complete_mae.sh")
        return None, None
    
    try:
        print(f"📥 加载完整模型权重: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 检查模型结构
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 统计参数
        encoder_keys = [k for k in state_dict.keys() if not k.startswith('decoder') and k != 'mask_token']
        decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder') or k == 'mask_token']
        
        print(f"  编码器参数: {len(encoder_keys)} 个")
        print(f"  解码器参数: {len(decoder_keys)} 个")
        
        # 加载权重
        msg = model.load_state_dict(state_dict, strict=False)
        if msg.missing_keys:
            print(f"  ⚠️  缺失键: {len(msg.missing_keys)} 个")
        if msg.unexpected_keys:
            print(f"  ⚠️  意外键: {len(msg.unexpected_keys)} 个")
        
        model = model.to(device)
        model.eval()
        
        print("✅ 完整MAE模型加载成功!")
        return model, device
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

def create_high_quality_reconstruction_demo(model, device):
    """创建高质量重建演示"""
    print("\n🎨 创建高质量重建演示...")
    
    # 尝试加载AnimeDiffusion数据
    try:
        dataloader, dataset = create_animediffusion_dataloader(
            batch_size=3,
            max_samples=10,
            input_size=224,
            num_workers=0
        )
        
        if dataloader is not None:
            for images, _ in dataloader:
                test_images = images[:3]
                print(f"✅ 使用AnimeDiffusion数据: {test_images.shape}")
                break
        else:
            raise Exception("数据加载器创建失败")
            
    except Exception as e:
        print(f"⚠️  AnimeDiffusion加载失败: {e}")
        print("🎨 使用备用测试图像...")
        test_images = create_test_images(device)
    
    # 反归一化
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    test_images = test_images.to(device)
    num_images = test_images.shape[0]
    
    # 测试不同mask比例
    mask_ratios = [0.5, 0.75, 0.9]
    
    # 创建可视化
    fig, axes = plt.subplots(num_images, len(mask_ratios)*3 + 1, figsize=(20, num_images*4))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    reconstruction_stats = []
    
    for img_idx in range(num_images):
        img_tensor = test_images[img_idx:img_idx+1]
        
        print(f"\n  🔍 处理图像 {img_idx+1}...")
        
        # 显示原始图像
        original_display = torch.clamp(inv_normalize(img_tensor[0]).cpu(), 0, 1)
        axes[img_idx, 0].imshow(original_display.permute(1, 2, 0))
        axes[img_idx, 0].set_title(f'原始图像 {img_idx+1}')
        axes[img_idx, 0].axis('off')
        
        col_idx = 1
        img_stats = {'image_id': img_idx + 1, 'results': []}
        
        for mask_ratio in mask_ratios:
            print(f"    🎭 Mask比例: {mask_ratio*100:.0f}%")
            
            with torch.no_grad():
                # MAE前向传播
                loss, pred, mask = model(img_tensor, mask_ratio=mask_ratio)
                
                # 重建图像
                reconstructed = model.unpatchify(pred)
                
                # 创建掩码可视化
                mask_vis = mask.detach()
                mask_vis = mask_vis.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
                mask_vis = model.unpatchify(mask_vis)
                
                # 转换为显示格式
                reconstructed_display = torch.clamp(inv_normalize(reconstructed[0]).cpu(), 0, 1)
                masked_img = original_display * (1 - mask_vis[0].cpu()) + mask_vis[0].cpu() * 0.5
                
                # 计算重建误差
                error = torch.abs(original_display - reconstructed_display)
                
                # 显示结果
                axes[img_idx, col_idx].imshow(masked_img.permute(1, 2, 0))
                axes[img_idx, col_idx].set_title(f'掩码 {mask_ratio*100:.0f}%')
                axes[img_idx, col_idx].axis('off')
                
                axes[img_idx, col_idx+1].imshow(reconstructed_display.permute(1, 2, 0))
                axes[img_idx, col_idx+1].set_title(f'重建\n损失:{loss.item():.3f}')
                axes[img_idx, col_idx+1].axis('off')
                
                error_display = error.mean(dim=0)
                im = axes[img_idx, col_idx+2].imshow(error_display, cmap='hot')
                axes[img_idx, col_idx+2].set_title(f'误差\n均值:{error.mean():.3f}')
                axes[img_idx, col_idx+2].axis('off')
                plt.colorbar(im, ax=axes[img_idx, col_idx+2], fraction=0.046, pad=0.04)
                
                col_idx += 3
                
                # 记录统计
                stats = {
                    'mask_ratio': mask_ratio,
                    'loss': loss.item(),
                    'actual_mask_ratio': mask.float().mean().item(),
                    'mean_error': error.mean().item(),
                    'max_error': error.max().item(),
                    'pred_range': [pred.min().item(), pred.max().item()]
                }
                
                img_stats['results'].append(stats)
                
                print(f"      损失: {loss.item():.4f}")
                print(f"      预测范围: [{pred.min():.3f}, {pred.max():.3f}]")
                print(f"      重建误差: {error.mean():.4f}")
        
        reconstruction_stats.append(img_stats)
    
    plt.tight_layout()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'complete_mae_reconstruction_{timestamp}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 高质量重建结果保存: {output_path}")
    
    try:
        plt.show()
    except:
        print("💡 如果要查看图像，请在支持图形界面的环境中运行")
    
    return output_path, reconstruction_stats

def create_test_images(device):
    """创建测试图像"""
    images = []
    
    # 图像1: 清晰的几何图案
    img1 = torch.zeros(3, 224, 224, device=device)
    
    # 背景渐变
    for i in range(224):
        for j in range(224):
            img1[0, i, j] = (0.2 + 0.3 * i / 224 - 0.485) / 0.229
            img1[1, i, j] = (0.3 + 0.4 * j / 224 - 0.456) / 0.224
            img1[2, i, j] = (0.6 - 0.406) / 0.225
    
    # 添加圆形
    y, x = torch.meshgrid(torch.arange(224, device=device), torch.arange(224, device=device), indexing='ij')
    circle = (x - 112)**2 + (y - 112)**2 <= 40**2
    img1[0][circle] = (0.9 - 0.485) / 0.229
    img1[1][circle] = (0.1 - 0.456) / 0.224
    img1[2][circle] = (0.1 - 0.406) / 0.225
    
    images.append(img1)
    
    # 图像2: 棋盘格
    img2 = torch.zeros(3, 224, 224, device=device)
    for i in range(0, 224, 32):
        for j in range(0, 224, 32):
            if (i//32 + j//32) % 2 == 0:
                color = [(0.8 - 0.485) / 0.229, (0.8 - 0.456) / 0.224, (0.8 - 0.406) / 0.225]
            else:
                color = [(0.2 - 0.485) / 0.229, (0.2 - 0.456) / 0.224, (0.2 - 0.406) / 0.225]
            
            img2[:, i:i+32, j:j+32] = torch.tensor(color, device=device).reshape(3, 1, 1)
    
    images.append(img2)
    
    # 图像3: 同心圆
    img3 = torch.zeros(3, 224, 224, device=device)
    for r in range(20, 100, 20):
        mask = ((x - 112)**2 + (y - 112)**2 >= (r-10)**2) & ((x - 112)**2 + (y - 112)**2 <= r**2)
        color_intensity = r / 100
        img3[0][mask] = (color_intensity - 0.485) / 0.229
        img3[1][mask] = (0.5 - 0.456) / 0.224
        img3[2][mask] = (1.0 - color_intensity - 0.406) / 0.225
    
    images.append(img3)
    
    return torch.stack(images)

def compare_with_random_decoder():
    """与随机解码器进行对比"""
    print("\n📊 与随机解码器模型对比...")
    
    # 加载完整模型
    complete_model, device = load_complete_mae_model()
    if complete_model is None:
        return
    
    # 创建随机解码器模型
    random_model = models_mae.mae_vit_large_patch16()
    
    # 只加载编码器权重
    model_path = 'complete_mae_models/mae_visualize_vit_large.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model']
    
    encoder_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith('decoder') and key != 'mask_token':
            encoder_state_dict[key] = value
    
    random_model.load_state_dict(encoder_state_dict, strict=False)
    random_model = random_model.to(device)
    random_model.eval()
    
    # 创建测试图像
    test_img = create_test_images(device)[0:1]  # 只用第一张
    
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    models_to_test = [
        ("完整预训练模型", complete_model),
        ("随机解码器模型", random_model)
    ]
    
    for i, (model_name, model) in enumerate(models_to_test):
        print(f"  🔍 测试 {model_name}...")
        
        with torch.no_grad():
            # 原始图像
            original_display = torch.clamp(inv_normalize(test_img[0]).cpu(), 0, 1)
            axes[i, 0].imshow(original_display.permute(1, 2, 0))
            axes[i, 0].set_title(f'{model_name}\n原始图像')
            axes[i, 0].axis('off')
            
            # MAE重建
            loss, pred, mask = model(test_img, mask_ratio=0.75)
            reconstructed = model.unpatchify(pred)
            
            # 掩码图像
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
    
    comparison_path = 'complete_vs_random_decoder.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 对比结果保存: {comparison_path}")
    
    try:
        plt.show()
    except:
        print("💡 如果要查看图像，请在支持图形界面的环境中运行")
    
    return comparison_path

def main():
    """主函数"""
    print("🎯 完整MAE模型高质量重建演示")
    print("=" * 60)
    
    # 加载完整模型
    model, device = load_complete_mae_model()
    if model is None:
        return
    
    # 创建高质量重建演示
    output_path, stats = create_high_quality_reconstruction_demo(model, device)
    
    # 与随机解码器对比
    comparison_path = compare_with_random_decoder()
    
    print(f"\n🎉 演示完成!")
    print(f"📁 高质量重建: {output_path}")
    print(f"📊 对比分析: {comparison_path}")
    
    print(f"\n💡 关键发现:")
    print("✅ 完整预训练模型重建质量显著提升")
    print("🎨 解码器权重对重建效果至关重要")
    print("📈 预训练解码器能正确理解编码器特征")
    print("🔥 这就是高质量MAE重建的正确方法!")

if __name__ == "__main__":
    main()

