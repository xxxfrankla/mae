#!/usr/bin/env python3
"""
动漫数据集 MAE 训练结果可视化
展示在真实动漫图片上的重建效果
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

class AnimeMAEVisualizer:
    def __init__(self, checkpoint_path=None, output_dir=None):
        """初始化动漫MAE可视化器"""
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"./anime_visualization_{timestamp}")
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(exist_ok=True)
        print(f"📁 结果保存到: {self.output_dir}")
        
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # 加载模型
        self.model = self._load_model(checkpoint_path)
        
        # 加载数据集
        self.dataset = self._load_anime_dataset()
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

    def _load_model(self, checkpoint_path):
        """加载训练好的模型"""
        print("🤖 加载MAE模型...")
        
        model = models_mae.mae_vit_base_patch16(norm_pix_loss=True)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['model'])
                epoch = checkpoint.get('epoch', 'unknown')
                print(f"✅ 加载训练模型成功 (epoch: {epoch})")
            except Exception as e:
                print(f"⚠️  加载checkpoint失败: {e}")
                print("使用随机初始化的模型")
        else:
            print("⚠️  使用随机初始化的模型")
        
        model.to(self.device)
        model.eval()
        return model

    def _load_anime_dataset(self):
        """加载动漫数据集"""
        print("🎌 加载动漫数据集...")
        try:
            ds = load_dataset("none-yet/anime-captions")
            print("✅ 动漫数据集加载成功")
            return ds['train']
        except Exception as e:
            print(f"❌ 动漫数据集加载失败: {e}")
            return None

    def visualize_anime_reconstruction(self, num_samples=8, mask_ratios=[0.5, 0.75, 0.9]):
        """可视化动漫图片的MAE重建效果"""
        print(f"🎨 可视化 {num_samples} 张动漫图片的重建效果...")
        
        if self.dataset is None:
            print("❌ 数据集未加载")
            return
        
        # 随机选择样本
        sample_indices = np.random.choice(len(self.dataset), num_samples, replace=False)
        
        for mask_ratio in mask_ratios:
            fig, axes = plt.subplots(3, num_samples, figsize=(num_samples*3, 9))
            
            print(f"\n🎭 掩码比例: {mask_ratio*100:.0f}%")
            
            for i, idx in enumerate(sample_indices):
                try:
                    # 获取原始图片
                    sample = self.dataset[int(idx)]
                    original_img = sample['image']
                    caption = sample['text'][:50] + "..." if len(sample['text']) > 50 else sample['text']
                    
                    # 确保是RGB模式
                    if original_img.mode != 'RGB':
                        original_img = original_img.convert('RGB')
                    
                    # 预处理
                    img_tensor = self.transform(original_img).unsqueeze(0).to(self.device)
                    
                    # MAE 前向传播
                    with torch.no_grad():
                        loss, pred, mask = self.model(img_tensor, mask_ratio=mask_ratio)
                        reconstructed = self.model.unpatchify(pred)
                        
                        # 创建掩码可视化
                        mask_vis = mask.detach()
                        mask_vis = mask_vis.unsqueeze(-1).repeat(1, 1, self.model.patch_embed.patch_size[0]**2 * 3)
                        mask_vis = self.model.unpatchify(mask_vis)
                    
                    # 转换为显示格式
                    original_display = torch.clamp(self.inv_normalize(img_tensor[0]).cpu(), 0, 1)
                    reconstructed_display = torch.clamp(self.inv_normalize(reconstructed[0]).cpu(), 0, 1)
                    masked_img = original_display * (1 - mask_vis[0].cpu()) + mask_vis[0].cpu() * 0.5
                    
                    # 显示结果
                    axes[0, i].imshow(original_display.permute(1, 2, 0))
                    axes[0, i].set_title(f'Original {i+1}', fontsize=10)
                    axes[0, i].axis('off')
                    
                    axes[1, i].imshow(masked_img.permute(1, 2, 0))
                    axes[1, i].set_title(f'Masked {i+1}', fontsize=10)
                    axes[1, i].axis('off')
                    
                    axes[2, i].imshow(reconstructed_display.permute(1, 2, 0))
                    axes[2, i].set_title(f'Reconstructed {i+1}\nLoss: {loss.item():.3f}', fontsize=10)
                    axes[2, i].axis('off')
                    
                    print(f"  样本 {i+1}: 损失 {loss.item():.4f}")
                    
                except Exception as e:
                    print(f"处理样本 {idx} 时出错: {e}")
                    for row in range(3):
                        axes[row, i].text(0.5, 0.5, f'Error\n{str(e)[:20]}', 
                                        ha='center', va='center')
                        axes[row, i].axis('off')
            
            plt.tight_layout()
            
            # 保存结果
            result_path = self.output_dir / f'anime_reconstruction_mask_{mask_ratio*100:.0f}percent.png'
            plt.savefig(result_path, dpi=150, bbox_inches='tight')
            print(f"✅ 保存: {result_path}")
            plt.close()

    def compare_with_synthetic(self):
        """对比动漫数据集和合成数据集的训练效果"""
        print(f"\n📊 对比动漫数据集和合成数据集的训练效果...")
        
        # 读取两个实验的日志
        anime_log = './output_anime/log.txt'
        synthetic_log = './output_m4/log.txt'
        
        def read_log(log_file):
            if not os.path.exists(log_file):
                return None
            
            epochs, losses = [], []
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        epochs.append(data['epoch'])
                        losses.append(data['train_loss'])
                    except:
                        continue
            return epochs, losses
        
        anime_data = read_log(anime_log)
        synthetic_data = read_log(synthetic_log)
        
        if anime_data is None and synthetic_data is None:
            print("❌ 未找到训练日志")
            return
        
        # 创建对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 损失对比
        if anime_data:
            epochs_a, losses_a = anime_data
            ax1.plot(epochs_a, losses_a, 'b-', linewidth=2, marker='o', 
                    label=f'Anime Dataset (final: {losses_a[-1]:.3f})', markersize=6)
        
        if synthetic_data:
            epochs_s, losses_s = synthetic_data
            ax1.plot(epochs_s, losses_s, 'r-', linewidth=2, marker='s', 
                    label=f'Synthetic Dataset (final: {losses_s[-1]:.3f})', markersize=6)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 数据集对比信息
        info_text = []
        if anime_data:
            info_text.append(f"🎌 Anime Dataset:")
            info_text.append(f"  • 337K real anime images")
            info_text.append(f"  • 512x512 → 224x224")
            info_text.append(f"  • Final loss: {losses_a[-1]:.4f}")
            info_text.append(f"  • Epochs: {len(epochs_a)}")
        
        if synthetic_data:
            info_text.append(f"\n🎨 Synthetic Dataset:")
            info_text.append(f"  • 250 synthetic images")
            info_text.append(f"  • 224x224 geometric patterns")
            info_text.append(f"  • Final loss: {losses_s[-1]:.4f}")
            info_text.append(f"  • Epochs: {len(epochs_s)}")
        
        ax2.text(0.05, 0.95, '\n'.join(info_text), transform=ax2.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax2.axis('off')
        ax2.set_title('Dataset Comparison')
        
        plt.tight_layout()
        
        # 保存对比图
        comparison_path = self.output_dir / 'dataset_comparison.png'
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"✅ 对比图保存: {comparison_path}")
        plt.close()

    def create_anime_mae_demo(self):
        """创建动漫MAE演示"""
        print(f"\n🎭 创建动漫MAE演示...")
        
        if self.dataset is None:
            return
        
        # 选择几张有代表性的动漫图片
        demo_indices = [0, 100, 200, 500, 1000, 1500]  # 不同风格的图片
        
        fig, axes = plt.subplots(len(demo_indices), 4, figsize=(16, len(demo_indices)*4))
        
        for i, idx in enumerate(demo_indices):
            try:
                # 获取图片
                sample = self.dataset[idx]
                original_img = sample['image']
                caption = sample['text']
                
                if original_img.mode != 'RGB':
                    original_img = original_img.convert('RGB')
                
                # 预处理
                img_tensor = self.transform(original_img).unsqueeze(0).to(self.device)
                
                # MAE处理
                with torch.no_grad():
                    loss, pred, mask = self.model(img_tensor, mask_ratio=0.75)
                    reconstructed = self.model.unpatchify(pred)
                    
                    # 掩码可视化
                    mask_vis = mask.detach()
                    mask_vis = mask_vis.unsqueeze(-1).repeat(1, 1, self.model.patch_embed.patch_size[0]**2 * 3)
                    mask_vis = self.model.unpatchify(mask_vis)
                
                # 显示格式转换
                original_display = torch.clamp(self.inv_normalize(img_tensor[0]).cpu(), 0, 1)
                reconstructed_display = torch.clamp(self.inv_normalize(reconstructed[0]).cpu(), 0, 1)
                masked_img = original_display * (1 - mask_vis[0].cpu()) + mask_vis[0].cpu() * 0.5
                
                # 计算重建误差
                error = torch.abs(original_display - reconstructed_display)
                error_display = error.mean(dim=0)
                
                # 显示
                axes[i, 0].imshow(original_display.permute(1, 2, 0))
                axes[i, 0].set_title(f'Original Anime {i+1}')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(masked_img.permute(1, 2, 0))
                axes[i, 1].set_title('75% Masked')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(reconstructed_display.permute(1, 2, 0))
                axes[i, 2].set_title(f'Reconstructed\nLoss: {loss.item():.3f}')
                axes[i, 2].axis('off')
                
                im = axes[i, 3].imshow(error_display, cmap='hot')
                axes[i, 3].set_title('Error Map')
                axes[i, 3].axis('off')
                plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
                
                print(f"  动漫图片 {i+1}: 损失 {loss.item():.4f}")
                
            except Exception as e:
                print(f"处理动漫图片 {idx} 时出错: {e}")
                for j in range(4):
                    axes[i, j].text(0.5, 0.5, f'Error: {str(e)[:30]}', 
                                  ha='center', va='center')
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        
        # 保存演示
        demo_path = self.output_dir / 'anime_mae_demo.png'
        plt.savefig(demo_path, dpi=150, bbox_inches='tight')
        print(f"✅ 动漫MAE演示保存: {demo_path}")
        plt.close()

    def analyze_anime_reconstruction_quality(self):
        """分析动漫图片重建质量"""
        print(f"\n🔍 分析动漫图片重建质量...")
        
        if self.dataset is None:
            return
        
        # 测试不同类型的动漫图片
        test_indices = np.random.choice(len(self.dataset), 50, replace=False)
        
        reconstruction_stats = []
        
        for idx in test_indices:
            try:
                sample = self.dataset[int(idx)]
                img = sample['image']
                caption = sample['text']
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    loss, pred, mask = self.model(img_tensor, mask_ratio=0.75)
                
                reconstruction_stats.append({
                    'index': idx,
                    'loss': loss.item(),
                    'caption_length': len(caption),
                    'caption': caption[:100]
                })
                
            except Exception as e:
                continue
        
        if not reconstruction_stats:
            print("❌ 没有成功处理的样本")
            return
        
        # 分析结果
        losses = [stat['loss'] for stat in reconstruction_stats]
        
        print(f"📊 重建质量分析 ({len(reconstruction_stats)} 个样本):")
        print(f"  平均损失: {np.mean(losses):.4f}")
        print(f"  损失标准差: {np.std(losses):.4f}")
        print(f"  最小损失: {np.min(losses):.4f}")
        print(f"  最大损失: {np.max(losses):.4f}")
        
        # 找出重建最好和最差的图片
        best_idx = np.argmin(losses)
        worst_idx = np.argmax(losses)
        
        print(f"\n🏆 重建最佳:")
        print(f"  损失: {reconstruction_stats[best_idx]['loss']:.4f}")
        print(f"  描述: {reconstruction_stats[best_idx]['caption']}")
        
        print(f"\n😅 重建最差:")
        print(f"  损失: {reconstruction_stats[worst_idx]['loss']:.4f}")
        print(f"  描述: {reconstruction_stats[worst_idx]['caption']}")
        
        # 保存统计信息
        stats_path = self.output_dir / 'anime_reconstruction_analysis.json'
        with open(stats_path, 'w') as f:
            json.dump({
                'summary': {
                    'num_samples': len(reconstruction_stats),
                    'mean_loss': float(np.mean(losses)),
                    'std_loss': float(np.std(losses)),
                    'min_loss': float(np.min(losses)),
                    'max_loss': float(np.max(losses))
                },
                'best_sample': reconstruction_stats[best_idx],
                'worst_sample': reconstruction_stats[worst_idx],
                'all_samples': reconstruction_stats
            }, f, indent=2)
        
        print(f"✅ 分析结果保存: {stats_path}")

def main():
    """主函数"""
    print("🎌 动漫数据集 MAE 结果可视化")
    print("=" * 60)
    
    # 查找最新的checkpoint
    checkpoint_path = './output_anime/checkpoint-2.pth'  # 最后一个epoch
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  未找到checkpoint: {checkpoint_path}")
        print("将使用随机初始化的模型进行演示")
        checkpoint_path = None
    
    # 创建可视化器
    visualizer = AnimeMAEVisualizer(checkpoint_path=checkpoint_path)
    
    # 1. 可视化重建效果
    visualizer.visualize_anime_reconstruction(num_samples=6, mask_ratios=[0.75])
    
    # 2. 创建演示
    visualizer.create_anime_mae_demo()
    
    # 3. 分析重建质量
    visualizer.analyze_anime_reconstruction_quality()
    
    # 4. 对比不同数据集
    visualizer.compare_with_synthetic()
    
    print(f"\n🎉 动漫数据集可视化完成!")
    print(f"📁 结果保存在: {visualizer.output_dir}")

if __name__ == "__main__":
    main()


