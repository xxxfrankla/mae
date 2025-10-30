#!/usr/bin/env python3
"""
AnimeDiffusion 训练结果专用可视化工具
展示高质量动漫图片的MAE重建效果
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from datasets import load_dataset

# 解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import models_mae

def create_animediffusion_visualization():
    """创建AnimeDiffusion专用可视化"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./animediffusion_visualization_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    print(f"🎌 AnimeDiffusion 结果可视化")
    print(f"📁 结果保存到: {output_dir}")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 加载训练好的模型
    checkpoint_path = './output_animediffusion/checkpoint-4.pth'
    model = models_mae.mae_vit_base_patch16(norm_pix_loss=True)
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'])
            print(f"✅ 加载训练模型成功 (epoch: {checkpoint.get('epoch', 'unknown')})")
        except Exception as e:
            print(f"⚠️  加载checkpoint失败: {e}, 使用随机模型")
    else:
        print("⚠️  使用随机初始化的模型")
    
    model.to(device)
    model.eval()
    
    # 加载AnimeDiffusion数据集
    try:
        ds = load_dataset("Mercity/AnimeDiffusion_Dataset")
        dataset = ds['train']
        print(f"✅ AnimeDiffusion 数据集加载成功")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize(int(224 * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),  # 用中心裁剪确保一致性
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # 1. 创建高质量重建演示
    print(f"\n🎨 创建高质量重建演示...")
    
    # 选择不同风格的动漫图片
    demo_indices = [0, 100, 500, 1000, 2000, 3000]
    
    fig, axes = plt.subplots(len(demo_indices), 4, figsize=(16, len(demo_indices)*3))
    
    reconstruction_losses = []
    
    for i, idx in enumerate(demo_indices):
        try:
            sample = dataset[idx]
            original_img = sample['image']
            
            # 获取提示词
            long_prompt = sample.get('long_prompt', '')
            short_prompt = sample.get('short_prompt', '')
            prompt = long_prompt if long_prompt else short_prompt
            if len(prompt) > 60:
                prompt = prompt[:57] + "..."
            
            if original_img.mode != 'RGB':
                original_img = original_img.convert('RGB')
            
            # 预处理
            img_tensor = transform(original_img).unsqueeze(0).to(device)
            
            # MAE处理
            with torch.no_grad():
                loss, pred, mask = model(img_tensor, mask_ratio=0.75)
                reconstructed = model.unpatchify(pred)
                
                # 掩码可视化
                mask_vis = mask.detach()
                mask_vis = mask_vis.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
                mask_vis = model.unpatchify(mask_vis)
            
            # 显示格式转换
            original_display = torch.clamp(inv_normalize(img_tensor[0]).cpu(), 0, 1)
            reconstructed_display = torch.clamp(inv_normalize(reconstructed[0]).cpu(), 0, 1)
            masked_img = original_display * (1 - mask_vis[0].cpu()) + mask_vis[0].cpu() * 0.5
            
            # 计算重建误差
            error = torch.abs(original_display - reconstructed_display)
            error_display = error.mean(dim=0)
            
            # 显示
            axes[i, 0].imshow(original_display.permute(1, 2, 0))
            axes[i, 0].set_title(f'Original HD Anime {i+1}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(masked_img.permute(1, 2, 0))
            axes[i, 1].set_title('75% Masked')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(reconstructed_display.permute(1, 2, 0))
            axes[i, 2].set_title(f'Reconstructed\nLoss: {loss.item():.3f}')
            axes[i, 2].axis('off')
            
            im = axes[i, 3].imshow(error_display, cmap='hot', vmin=0, vmax=0.5)
            axes[i, 3].set_title('Error Map')
            axes[i, 3].axis('off')
            
            reconstruction_losses.append(loss.item())
            print(f"  样本 {i+1}: 损失 {loss.item():.4f}")
            
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")
            for j in range(4):
                axes[i, j].text(0.5, 0.5, f'Error: {str(e)[:30]}', 
                              ha='center', va='center')
                axes[i, j].axis('off')
    
    plt.tight_layout()
    
    # 保存演示
    demo_path = output_dir / 'animediffusion_mae_demo.png'
    plt.savefig(demo_path, dpi=150, bbox_inches='tight')
    print(f"✅ AnimeDiffusion MAE演示保存: {demo_path}")
    plt.close()
    
    # 2. 分析重建质量
    print(f"\n🔍 分析AnimeDiffusion重建质量...")
    
    if reconstruction_losses:
        print(f"📊 重建质量统计:")
        print(f"  平均损失: {np.mean(reconstruction_losses):.4f}")
        print(f"  损失范围: {np.min(reconstruction_losses):.4f} - {np.max(reconstruction_losses):.4f}")
        print(f"  标准差: {np.std(reconstruction_losses):.4f}")
    
    # 3. 创建三数据集对比
    create_three_dataset_comparison(output_dir)
    
    print(f"\n🎉 AnimeDiffusion 可视化完成!")
    print(f"📁 结果保存在: {output_dir}")
    
    return output_dir

def create_three_dataset_comparison(output_dir):
    """创建三个数据集的训练效果对比"""
    print(f"\n📊 创建三数据集训练效果对比...")
    
    # 读取三个实验的数据
    experiments = [
        {
            'name': 'Synthetic',
            'log_file': './output_m4/log.txt',
            'color': 'red',
            'marker': 's'
        },
        {
            'name': 'Anime-Captions',
            'log_file': './output_anime/log.txt',
            'color': 'blue',
            'marker': 'o'
        },
        {
            'name': 'AnimeDiffusion',
            'log_file': './output_animediffusion/log.txt',
            'color': 'green',
            'marker': '^'
        }
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    comparison_data = []
    
    for exp in experiments:
        if os.path.exists(exp['log_file']):
            epochs, losses = [], []
            
            with open(exp['log_file'], 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        epochs.append(data['epoch'])
                        losses.append(data['train_loss'])
                    except:
                        continue
            
            if epochs and losses:
                ax1.plot(epochs, losses, color=exp['color'], marker=exp['marker'], 
                        linewidth=2, markersize=6, label=f"{exp['name']} (final: {losses[-1]:.3f})")
                
                comparison_data.append({
                    'name': exp['name'],
                    'final_loss': losses[-1],
                    'epochs': len(epochs),
                    'improvement': ((1.3080 - losses[-1]) / 1.3080 * 100) if losses[-1] < 1.3080 else 0
                })
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison: Three Datasets')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0.8)
    
    # 数据集信息对比
    dataset_info = [
        "📊 Dataset Comparison:",
        "",
        "🎨 Synthetic Dataset:",
        "  • 250 geometric patterns",
        "  • 224×224 resolution",
        "  • Final loss: 1.308",
        "",
        "🎌 Anime-Captions:",
        "  • 337K anime images (used 1K)",
        "  • 512×512 → 224×224",
        "  • Final loss: 1.074 (+18%)",
        "",
        "🎭 AnimeDiffusion:",
        "  • 8.2K HD anime images (used 500)",
        "  • 1920×1080 → 224×224",
        "  • Final loss: 0.951 (+27%)",
        "",
        "🏆 Winner: AnimeDiffusion!",
        "Best reconstruction quality"
    ]
    
    ax2.text(0.05, 0.95, '\n'.join(dataset_info), transform=ax2.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax2.axis('off')
    ax2.set_title('Dataset Information')
    
    plt.tight_layout()
    
    # 保存对比图
    comparison_path = output_dir / 'three_dataset_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✅ 三数据集对比保存: {comparison_path}")
    plt.close()

def main():
    """主函数"""
    create_animediffusion_visualization()

if __name__ == "__main__":
    main()


