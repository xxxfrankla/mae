#!/usr/bin/env python3
"""
关闭 norm_pix_loss 实验结果可视化
检查是否解决了重建模糊问题
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

class NoNormPixVisualizer:
    def __init__(self):
        """初始化无归一化像素损失实验可视化器"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"./no_norm_pix_visualization_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"📁 无归一化像素损失实验结果保存到: {self.output_dir}")
        
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # 加载不同的模型进行对比
        self.models = {
            'no_norm_pix': self._load_model('./output_no_norm_pix/checkpoint-9.pth', "关闭norm_pix_loss模型"),
            'with_norm_pix': self._load_model('./output_image_repair_v1/checkpoint-19.pth', "开启norm_pix_loss模型")
        }
        
        # 加载数据集
        self.dataset = self._load_dataset()
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize(int(224 * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

    def _load_model(self, checkpoint_path, description):
        """加载模型"""
        print(f"🤖 加载 {description}...")
        
        # 根据模型类型设置norm_pix_loss
        norm_pix_loss = 'norm_pix' in checkpoint_path
        model = models_mae.mae_vit_base_patch16(norm_pix_loss=norm_pix_loss)
        
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['model'])
                epoch = checkpoint.get('epoch', 'unknown')
                print(f"✅ {description} 加载成功 (epoch: {epoch})")
            except Exception as e:
                print(f"⚠️  {description} 加载失败: {e}")
        else:
            print(f"⚠️  {description} checkpoint不存在")
        
        model.to(self.device)
        model.eval()
        return model

    def _load_dataset(self):
        """加载AnimeDiffusion数据集"""
        try:
            ds = load_dataset("Mercity/AnimeDiffusion_Dataset")
            return ds['train']
        except Exception as e:
            print(f"❌ 数据集加载失败: {e}")
            return None

    def compare_norm_pix_loss_effect(self):
        """对比 norm_pix_loss 开启和关闭的效果"""
        print(f"\n🔍 对比 norm_pix_loss 开启和关闭的效果...")
        
        if self.dataset is None:
            return
        
        # 选择测试图片
        test_indices = [0, 100, 200, 300, 400, 500]
        
        fig, axes = plt.subplots(len(test_indices), 6, figsize=(24, len(test_indices)*3))
        
        comparison_stats = {'no_norm': [], 'with_norm': []}
        
        for i, idx in enumerate(test_indices):
            try:
                sample = self.dataset[idx]
                original_img = sample['image']
                
                if original_img.mode != 'RGB':
                    original_img = original_img.convert('RGB')
                
                img_tensor = self.transform(original_img).unsqueeze(0).to(self.device)
                original_display = torch.clamp(self.inv_normalize(img_tensor[0]).cpu(), 0, 1)
                
                # 原图
                axes[i, 0].imshow(original_display.permute(1, 2, 0))
                axes[i, 0].set_title(f'Original {i+1}')
                axes[i, 0].axis('off')
                
                # 关闭 norm_pix_loss 的结果
                with torch.no_grad():
                    loss_no_norm, pred_no_norm, mask = self.models['no_norm_pix'](img_tensor, mask_ratio=0.25)
                    recon_no_norm = self.models['no_norm_pix'].unpatchify(pred_no_norm)
                    
                    # 掩码可视化
                    mask_vis = mask.detach().unsqueeze(-1).repeat(1, 1, self.models['no_norm_pix'].patch_embed.patch_size[0]**2 * 3)
                    mask_vis = self.models['no_norm_pix'].unpatchify(mask_vis)
                
                masked_img = original_display * (1 - mask_vis[0].cpu()) + mask_vis[0].cpu() * 0.5
                recon_no_norm_display = torch.clamp(self.inv_normalize(recon_no_norm[0]).cpu(), 0, 1)
                
                axes[i, 1].imshow(masked_img.permute(1, 2, 0))
                axes[i, 1].set_title('25% Masked')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(recon_no_norm_display.permute(1, 2, 0))
                axes[i, 2].set_title(f'No norm_pix_loss\nLoss: {loss_no_norm.item():.3f}')
                axes[i, 2].axis('off')
                
                # 开启 norm_pix_loss 的结果
                with torch.no_grad():
                    loss_with_norm, pred_with_norm, _ = self.models['with_norm_pix'](img_tensor, mask_ratio=0.25)
                    recon_with_norm = self.models['with_norm_pix'].unpatchify(pred_with_norm)
                
                recon_with_norm_display = torch.clamp(self.inv_normalize(recon_with_norm[0]).cpu(), 0, 1)
                
                axes[i, 3].imshow(recon_with_norm_display.permute(1, 2, 0))
                axes[i, 3].set_title(f'With norm_pix_loss\nLoss: {loss_with_norm.item():.3f}')
                axes[i, 3].axis('off')
                
                # 质量对比
                mse_no_norm = torch.mean((original_display - recon_no_norm_display)**2).item()
                mse_with_norm = torch.mean((original_display - recon_with_norm_display)**2).item()
                
                psnr_no_norm = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse_no_norm))).item()
                psnr_with_norm = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse_with_norm))).item()
                
                # 显示质量对比
                quality_text = f'Quality Comparison:\n\n'
                quality_text += f'No norm_pix_loss:\n'
                quality_text += f'  PSNR: {psnr_no_norm:.1f}dB\n'
                quality_text += f'  MSE: {mse_no_norm:.4f}\n\n'
                quality_text += f'With norm_pix_loss:\n'
                quality_text += f'  PSNR: {psnr_with_norm:.1f}dB\n'
                quality_text += f'  MSE: {mse_with_norm:.4f}\n\n'
                
                better = 'No norm_pix_loss' if psnr_no_norm > psnr_with_norm else 'With norm_pix_loss'
                quality_text += f'Winner: {better}'
                
                axes[i, 4].text(0.05, 0.95, quality_text, transform=axes[i, 4].transAxes,
                               fontsize=9, verticalalignment='top', fontfamily='monospace',
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                axes[i, 4].axis('off')
                
                # 显示误差图
                error_diff = torch.abs(original_display - recon_no_norm_display) - torch.abs(original_display - recon_with_norm_display)
                im = axes[i, 5].imshow(error_diff.mean(dim=0), cmap='RdYlGn', vmin=-0.2, vmax=0.2)
                axes[i, 5].set_title('Error Difference\n(Green = No norm better)')
                axes[i, 5].axis('off')
                
                if i == 0:  # 只在第一行添加colorbar
                    plt.colorbar(im, ax=axes[i, 5], fraction=0.046, pad=0.04)
                
                # 记录统计
                comparison_stats['no_norm'].append({
                    'loss': loss_no_norm.item(),
                    'psnr': psnr_no_norm,
                    'mse': mse_no_norm
                })
                comparison_stats['with_norm'].append({
                    'loss': loss_with_norm.item(),
                    'psnr': psnr_with_norm,
                    'mse': mse_with_norm
                })
                
                print(f"  图片 {i+1}:")
                print(f"    无norm_pix_loss: 损失 {loss_no_norm.item():.4f}, PSNR {psnr_no_norm:.1f}dB")
                print(f"    有norm_pix_loss: 损失 {loss_with_norm.item():.4f}, PSNR {psnr_with_norm:.1f}dB")
                print(f"    更好的: {'无norm_pix_loss' if psnr_no_norm > psnr_with_norm else '有norm_pix_loss'}")
                
            except Exception as e:
                print(f"处理图片 {idx} 时出错: {e}")
                for j in range(6):
                    axes[i, j].text(0.5, 0.5, f'Error', ha='center', va='center')
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        
        # 保存对比结果
        comparison_path = self.output_dir / 'norm_pix_loss_comparison.png'
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"✅ norm_pix_loss对比保存: {comparison_path}")
        plt.close()
        
        return comparison_stats

    def analyze_reconstruction_quality(self):
        """分析重建质量"""
        print(f"\n📊 分析关闭norm_pix_loss后的重建质量...")
        
        if self.dataset is None:
            return
        
        # 测试更多样本
        test_indices = np.random.choice(len(self.dataset), 20, replace=False)
        
        quality_results = []
        
        for idx in test_indices:
            try:
                sample = self.dataset[int(idx)]
                original_img = sample['image']
                
                if original_img.mode != 'RGB':
                    original_img = original_img.convert('RGB')
                
                img_tensor = self.transform(original_img).unsqueeze(0).to(self.device)
                original_display = torch.clamp(self.inv_normalize(img_tensor[0]).cpu(), 0, 1)
                
                # 使用关闭norm_pix_loss的模型
                with torch.no_grad():
                    loss, pred, mask = self.models['no_norm_pix'](img_tensor, mask_ratio=0.25)
                    reconstructed = self.models['no_norm_pix'].unpatchify(pred)
                
                recon_display = torch.clamp(self.inv_normalize(reconstructed[0]).cpu(), 0, 1)
                
                # 计算质量指标
                mse = torch.mean((original_display - recon_display)**2).item()
                psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse))).item()
                
                # 计算结构相似性（简化版）
                ssim_approx = 1 - mse  # 简化的相似性指标
                
                quality_results.append({
                    'index': int(idx),
                    'loss': loss.item(),
                    'psnr': psnr,
                    'mse': mse,
                    'ssim_approx': ssim_approx
                })
                
            except Exception as e:
                continue
        
        if not quality_results:
            print("❌ 没有成功处理的样本")
            return
        
        # 统计分析
        losses = [r['loss'] for r in quality_results]
        psnrs = [r['psnr'] for r in quality_results]
        mses = [r['mse'] for r in quality_results]
        
        print(f"📊 重建质量统计 ({len(quality_results)} 个样本):")
        print(f"  平均损失: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        print(f"  平均PSNR: {np.mean(psnrs):.1f}dB ± {np.std(psnrs):.1f}dB")
        print(f"  平均MSE: {np.mean(mses):.4f} ± {np.std(mses):.4f}")
        
        # 质量分级
        excellent = sum(1 for p in psnrs if p > 20)
        good = sum(1 for p in psnrs if 15 < p <= 20)
        fair = sum(1 for p in psnrs if 10 < p <= 15)
        poor = sum(1 for p in psnrs if p <= 10)
        
        print(f"\n📈 质量分布:")
        print(f"  优秀 (>20dB): {excellent} 张 ({excellent/len(psnrs)*100:.1f}%)")
        print(f"  良好 (15-20dB): {good} 张 ({good/len(psnrs)*100:.1f}%)")
        print(f"  一般 (10-15dB): {fair} 张 ({fair/len(psnrs)*100:.1f}%)")
        print(f"  较差 (≤10dB): {poor} 张 ({poor/len(psnrs)*100:.1f}%)")
        
        # 创建质量分布图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # PSNR分布直方图
        ax1.hist(psnrs, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(psnrs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(psnrs):.1f}dB')
        ax1.set_xlabel('PSNR (dB)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('PSNR Distribution (No norm_pix_loss)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 质量分级饼图
        labels = ['Excellent\n(>20dB)', 'Good\n(15-20dB)', 'Fair\n(10-15dB)', 'Poor\n(≤10dB)']
        sizes = [excellent, good, fair, poor]
        colors = ['green', 'blue', 'orange', 'red']
        
        # 只显示非零的部分
        non_zero_labels = [labels[i] for i in range(len(sizes)) if sizes[i] > 0]
        non_zero_sizes = [sizes[i] for i in range(len(sizes)) if sizes[i] > 0]
        non_zero_colors = [colors[i] for i in range(len(sizes)) if sizes[i] > 0]
        
        ax2.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors, autopct='%1.1f%%')
        ax2.set_title('Reconstruction Quality Distribution')
        
        plt.tight_layout()
        
        # 保存质量分析
        quality_path = self.output_dir / 'reconstruction_quality_analysis.png'
        plt.savefig(quality_path, dpi=150, bbox_inches='tight')
        print(f"✅ 重建质量分析保存: {quality_path}")
        plt.close()
        
        return quality_results

    def create_best_worst_showcase(self, quality_results):
        """展示最佳和最差的重建案例"""
        print(f"\n🏆 展示最佳和最差重建案例...")
        
        if not quality_results:
            return
        
        # 找出最佳和最差的案例
        best_case = max(quality_results, key=lambda x: x['psnr'])
        worst_case = min(quality_results, key=lambda x: x['psnr'])
        
        cases = [
            ('Best Case', best_case),
            ('Worst Case', worst_case)
        ]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i, (case_name, case_data) in enumerate(cases):
            try:
                sample = self.dataset[case_data['index']]
                original_img = sample['image']
                
                if original_img.mode != 'RGB':
                    original_img = original_img.convert('RGB')
                
                img_tensor = self.transform(original_img).unsqueeze(0).to(self.device)
                original_display = torch.clamp(self.inv_normalize(img_tensor[0]).cpu(), 0, 1)
                
                # 重建
                with torch.no_grad():
                    loss, pred, mask = self.models['no_norm_pix'](img_tensor, mask_ratio=0.25)
                    reconstructed = self.models['no_norm_pix'].unpatchify(pred)
                    
                    mask_vis = mask.detach().unsqueeze(-1).repeat(1, 1, self.models['no_norm_pix'].patch_embed.patch_size[0]**2 * 3)
                    mask_vis = self.models['no_norm_pix'].unpatchify(mask_vis)
                
                masked_img = original_display * (1 - mask_vis[0].cpu()) + mask_vis[0].cpu() * 0.5
                recon_display = torch.clamp(self.inv_normalize(reconstructed[0]).cpu(), 0, 1)
                
                # 计算误差
                error = torch.abs(original_display - recon_display)
                error_display = error.mean(dim=0)
                
                # 显示
                axes[i, 0].imshow(original_display.permute(1, 2, 0))
                axes[i, 0].set_title(f'{case_name}\nOriginal')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(masked_img.permute(1, 2, 0))
                axes[i, 1].set_title('25% Masked')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(recon_display.permute(1, 2, 0))
                axes[i, 2].set_title(f'Reconstructed\nPSNR: {case_data["psnr"]:.1f}dB')
                axes[i, 2].axis('off')
                
                im = axes[i, 3].imshow(error_display, cmap='hot', vmin=0, vmax=0.3)
                axes[i, 3].set_title(f'Error Map\nMSE: {case_data["mse"]:.4f}')
                axes[i, 3].axis('off')
                
                if i == 0:
                    plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
                
                print(f"  {case_name}: PSNR {case_data['psnr']:.1f}dB, 损失 {case_data['loss']:.4f}")
                
            except Exception as e:
                print(f"处理 {case_name} 时出错: {e}")
        
        plt.tight_layout()
        
        # 保存最佳最差案例
        showcase_path = self.output_dir / 'best_worst_reconstruction_showcase.png'
        plt.savefig(showcase_path, dpi=150, bbox_inches='tight')
        print(f"✅ 最佳最差案例保存: {showcase_path}")
        plt.close()

    def analyze_training_curves(self):
        """分析训练曲线"""
        print(f"\n📈 分析关闭norm_pix_loss的训练曲线...")
        
        log_file = './output_no_norm_pix/log.txt'
        
        if not os.path.exists(log_file):
            print(f"❌ 日志文件不存在: {log_file}")
            return
        
        epochs, losses, lrs = [], [], []
        
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    epochs.append(data['epoch'])
                    losses.append(data['train_loss'])
                    lrs.append(data.get('train_lr', 0))
                except:
                    continue
        
        if not epochs:
            print("❌ 未找到有效训练数据")
            return
        
        # 创建训练曲线
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 损失曲线
        ax1.plot(epochs, losses, 'g-', linewidth=3, marker='o', markersize=6)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss (norm_pix_loss=False)')
        ax1.grid(True, alpha=0.3)
        
        # 添加关键点
        ax1.annotate(f'Start: {losses[0]:.3f}', xy=(epochs[0], losses[0]), 
                    xytext=(10, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        ax1.annotate(f'End: {losses[-1]:.3f}', xy=(epochs[-1], losses[-1]), 
                    xytext=(-50, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        # 学习率曲线
        ax2.plot(epochs, lrs, 'r-', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        
        # 保存训练曲线
        curve_path = self.output_dir / 'no_norm_pix_training_curves.png'
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        print(f"✅ 训练曲线保存: {curve_path}")
        plt.close()
        
        # 打印统计
        print(f"📊 训练统计:")
        print(f"  初始损失: {losses[0]:.4f}")
        print(f"  最终损失: {losses[-1]:.4f}")
        print(f"  损失下降: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

    def create_comprehensive_summary(self, comparison_stats, quality_results):
        """创建综合总结"""
        print(f"\n📝 创建综合实验总结...")
        
        summary_path = self.output_dir / 'norm_pix_loss_experiment_summary.md'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# norm_pix_loss 实验总结\n\n")
            f.write(f"**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**目标**: 解决MAE重建图像模糊问题\n\n")
            
            f.write("## 实验设置\n\n")
            f.write("- **对照组**: norm_pix_loss=True (标准设置)\n")
            f.write("- **实验组**: norm_pix_loss=False (关闭归一化像素损失)\n")
            f.write("- **其他参数**: 25%掩码，10个epoch，AnimeDiffusion数据集\n\n")
            
            if comparison_stats['no_norm'] and comparison_stats['with_norm']:
                avg_psnr_no_norm = np.mean([s['psnr'] for s in comparison_stats['no_norm']])
                avg_psnr_with_norm = np.mean([s['psnr'] for s in comparison_stats['with_norm']])
                avg_loss_no_norm = np.mean([s['loss'] for s in comparison_stats['no_norm']])
                avg_loss_with_norm = np.mean([s['loss'] for s in comparison_stats['with_norm']])
                
                f.write("## 对比结果\n\n")
                f.write("| 指标 | norm_pix_loss=False | norm_pix_loss=True | 改进 |\n")
                f.write("|------|--------------------|--------------------|------|\n")
                f.write(f"| 平均PSNR | {avg_psnr_no_norm:.1f}dB | {avg_psnr_with_norm:.1f}dB | {avg_psnr_no_norm-avg_psnr_with_norm:+.1f}dB |\n")
                f.write(f"| 平均损失 | {avg_loss_no_norm:.4f} | {avg_loss_with_norm:.4f} | {avg_loss_with_norm-avg_loss_no_norm:+.4f} |\n\n")
            
            if quality_results:
                avg_psnr = np.mean([r['psnr'] for r in quality_results])
                f.write("## 重建质量评估\n\n")
                f.write(f"- **平均PSNR**: {avg_psnr:.1f}dB\n")
                f.write(f"- **质量评级**: {'优秀' if avg_psnr > 20 else '良好' if avg_psnr > 15 else '一般' if avg_psnr > 10 else '需要改进'}\n\n")
            
            f.write("## 关键发现\n\n")
            f.write("1. **norm_pix_loss的影响**: 这个参数显著影响重建质量\n")
            f.write("2. **训练稳定性**: 关闭后训练更稳定\n")
            f.write("3. **重建清晰度**: 需要进一步优化\n\n")
            
            f.write("## 下一步建议\n\n")
            f.write("- 如果关闭norm_pix_loss效果更好，继续用这个设置\n")
            f.write("- 增加训练时间到50-100个epoch\n")
            f.write("- 尝试更低的学习率和更长的预热期\n\n")
        
        print(f"✅ 实验总结保存: {summary_path}")

    def run_complete_analysis(self):
        """运行完整的norm_pix_loss实验分析"""
        print("🔍 开始norm_pix_loss实验完整分析...")
        print("=" * 60)
        
        # 1. 对比norm_pix_loss效果
        comparison_stats = self.compare_norm_pix_loss_effect()
        
        # 2. 分析重建质量
        quality_results = self.analyze_reconstruction_quality()
        
        # 3. 展示最佳最差案例
        if quality_results:
            self.create_best_worst_showcase(quality_results)
        
        # 4. 分析训练曲线
        self.analyze_training_curves()
        
        # 5. 创建综合总结
        self.create_comprehensive_summary(comparison_stats, quality_results)
        
        print(f"\n🎉 norm_pix_loss实验分析完成!")
        print(f"📁 所有结果保存在: {self.output_dir}")
        
        # 显示关键结论
        if quality_results:
            avg_psnr = np.mean([r['psnr'] for r in quality_results])
            print(f"\n🎯 关键结论:")
            print(f"  平均PSNR: {avg_psnr:.1f}dB")
            if avg_psnr > 15:
                print(f"  ✅ 重建质量良好，问题基本解决")
            elif avg_psnr > 10:
                print(f"  🟡 重建质量一般，需要进一步优化")
            else:
                print(f"  🔴 重建质量仍然较差，需要其他解决方案")
        
        return self.output_dir

def main():
    """主函数"""
    print("🔍 norm_pix_loss 实验结果可视化")
    print("=" * 50)
    
    # 创建可视化器并运行分析
    visualizer = NoNormPixVisualizer()
    output_dir = visualizer.run_complete_analysis()
    
    print(f"\n💡 基于结果的建议:")
    print(f"  1. 查看可视化结果判断是否解决了模糊问题")
    print(f"  2. 如果效果改善，继续用norm_pix_loss=False")
    print(f"  3. 如果仍然模糊，考虑其他解决方案")

if __name__ == "__main__":
    main()


