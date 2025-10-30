#!/usr/bin/env python3
"""
有组织的 MAE 可视化工具
每次运行都创建新的文件夹保存结果
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

# 解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import models_mae

class MAEVisualizer:
    def __init__(self, base_output_dir='./visualization_results'):
        """初始化可视化器"""
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # 创建带时间戳的文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.base_output_dir / f"mae_results_{timestamp}"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"📁 结果将保存到: {self.output_dir}")
        
        # 创建子文件夹
        self.curves_dir = self.output_dir / "training_curves"
        self.reconstruction_dir = self.output_dir / "reconstructions"
        self.analysis_dir = self.output_dir / "analysis"
        self.samples_dir = self.output_dir / "dataset_samples"
        
        for dir_path in [self.curves_dir, self.reconstruction_dir, self.analysis_dir, self.samples_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")

    def analyze_training_log(self, log_file='./output_m4/log.txt'):
        """分析训练日志并生成曲线"""
        print("📊 分析训练日志...")
        
        if not os.path.exists(log_file):
            print(f"❌ 日志文件不存在: {log_file}")
            return None
        
        epochs = []
        losses = []
        lrs = []
        
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
            print("❌ 未找到有效的训练数据")
            return None
        
        # 生成训练曲线
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 损失曲线
        ax1.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('MAE Training Loss Curve')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # 添加数值标注
        for i, (epoch, loss) in enumerate(zip(epochs, losses)):
            if i % max(1, len(epochs)//5) == 0:  # 每5个点标注一次
                ax1.annotate(f'{loss:.3f}', (epoch, loss), 
                           textcoords="offset points", xytext=(0,10), ha='center')
        
        # 学习率曲线
        ax2.plot(epochs, lrs, 'r-', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        
        # 保存图片
        curve_path = self.curves_dir / 'training_curves.png'
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        print(f"✅ 训练曲线保存: {curve_path}")
        plt.close()
        
        # 保存统计信息
        stats = {
            'total_epochs': len(epochs),
            'initial_loss': losses[0],
            'final_loss': losses[-1],
            'loss_reduction': ((losses[0] - losses[-1]) / losses[0] * 100),
            'max_lr': max(lrs),
            'min_lr': min(lrs)
        }
        
        stats_path = self.analysis_dir / 'training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📈 训练统计:")
        print(f"  训练轮数: {stats['total_epochs']}")
        print(f"  初始损失: {stats['initial_loss']:.4f}")
        print(f"  最终损失: {stats['final_loss']:.4f}")
        print(f"  损失下降: {stats['loss_reduction']:.1f}%")
        print(f"  最高学习率: {stats['max_lr']:.2e}")
        
        return stats

    def show_dataset_samples(self, test_dir='./test_dataset/train'):
        """展示数据集样本"""
        print("🖼️  展示测试数据集样本...")
        
        if not os.path.exists(test_dir):
            print(f"❌ 测试数据集不存在: {test_dir}")
            return
        
        # 收集样本
        class_samples = []
        class_names = sorted(os.listdir(test_dir))
        
        for class_name in class_names:
            class_path = os.path.join(test_dir, class_name)
            if os.path.isdir(class_path):
                img_files = sorted(os.listdir(class_path))
                if img_files:
                    # 每个类别取前3张图片
                    for i, img_file in enumerate(img_files[:3]):
                        img_path = os.path.join(class_path, img_file)
                        class_samples.append((class_name, img_path, i))
        
        if not class_samples:
            print("❌ 未找到测试图片")
            return
        
        # 按类别组织样本
        classes = {}
        for class_name, img_path, idx in class_samples:
            if class_name not in classes:
                classes[class_name] = []
            classes[class_name].append(img_path)
        
        # 创建网格可视化
        n_classes = len(classes)
        max_samples = max(len(samples) for samples in classes.values())
        
        fig, axes = plt.subplots(n_classes, max_samples, figsize=(max_samples*3, n_classes*3))
        
        if n_classes == 1:
            axes = axes.reshape(1, -1)
        elif max_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (class_name, img_paths) in enumerate(classes.items()):
            for j, img_path in enumerate(img_paths):
                try:
                    img = Image.open(img_path).convert('RGB')
                    axes[i, j].imshow(img)
                    axes[i, j].set_title(f'{class_name} - {j+1}')
                    axes[i, j].axis('off')
                except Exception as e:
                    print(f"加载图片失败 {img_path}: {e}")
            
            # 填充空白位置
            for j in range(len(img_paths), max_samples):
                axes[i, j].axis('off')
        
        plt.tight_layout()
        
        # 保存结果
        samples_path = self.samples_dir / 'dataset_samples_grid.png'
        plt.savefig(samples_path, dpi=150, bbox_inches='tight')
        print(f"✅ 数据集样本保存: {samples_path}")
        plt.close()
        
        # 单独保存每个类别的第一张图片
        for class_name, img_paths in classes.items():
            if img_paths:
                try:
                    img = Image.open(img_paths[0]).convert('RGB')
                    class_path = self.samples_dir / f'{class_name}_sample.png'
                    img.save(class_path)
                except:
                    pass

    def visualize_mae_reconstruction(self, test_img_path=None):
        """可视化 MAE 重建过程"""
        print("🎨 生成 MAE 重建可视化...")
        
        # 创建模型
        model = models_mae.mae_vit_base_patch16(norm_pix_loss=True)
        model.to(self.device)
        model.eval()
        
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        
        # 找测试图片
        if test_img_path is None:
            test_img_path = './test_dataset/train/class_00/img_0000.png'
        
        if not os.path.exists(test_img_path):
            print(f"❌ 测试图片不存在: {test_img_path}")
            return
        
        # 加载图片
        original_img = Image.open(test_img_path).convert('RGB')
        img_tensor = transform(original_img).unsqueeze(0).to(self.device)
        
        # 测试不同掩码比例
        mask_ratios = [0.25, 0.5, 0.75, 0.9]
        
        fig, axes = plt.subplots(len(mask_ratios), 4, figsize=(16, len(mask_ratios)*4))
        
        reconstruction_stats = {}
        
        for i, mask_ratio in enumerate(mask_ratios):
            with torch.no_grad():
                loss, pred, mask = model(img_tensor, mask_ratio=mask_ratio)
                reconstructed = model.unpatchify(pred)
                
                # 创建掩码可视化
                mask_vis = mask.detach()
                mask_vis = mask_vis.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
                mask_vis = model.unpatchify(mask_vis)
            
            # 转换为显示格式
            original_display = torch.clamp(inv_normalize(img_tensor[0]).cpu(), 0, 1)
            reconstructed_display = torch.clamp(inv_normalize(reconstructed[0]).cpu(), 0, 1)
            masked_img = original_display * (1 - mask_vis[0].cpu()) + mask_vis[0].cpu() * 0.5
            
            # 计算重建误差
            error = torch.abs(original_display - reconstructed_display)
            error_display = error.mean(dim=0)
            
            # 显示结果
            axes[i, 0].imshow(original_display.permute(1, 2, 0))
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(masked_img.permute(1, 2, 0))
            axes[i, 1].set_title(f'Masked ({mask_ratio*100:.0f}%)')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(reconstructed_display.permute(1, 2, 0))
            axes[i, 2].set_title(f'Reconstructed\nLoss: {loss.item():.3f}')
            axes[i, 2].axis('off')
            
            im = axes[i, 3].imshow(error_display, cmap='hot')
            axes[i, 3].set_title('Reconstruction Error')
            axes[i, 3].axis('off')
            plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
            
            # 保存统计信息
            reconstruction_stats[f'mask_{mask_ratio}'] = {
                'loss': loss.item(),
                'mask_ratio': mask_ratio,
                'mean_error': error.mean().item(),
                'max_error': error.max().item()
            }
            
            print(f"  掩码比例 {mask_ratio*100:.0f}%: 损失 {loss.item():.4f}, 平均误差 {error.mean().item():.4f}")
        
        plt.tight_layout()
        
        # 保存重建结果
        recon_path = self.reconstruction_dir / 'mae_reconstruction_comparison.png'
        plt.savefig(recon_path, dpi=150, bbox_inches='tight')
        print(f"✅ MAE 重建对比保存: {recon_path}")
        plt.close()
        
        # 保存统计信息
        stats_path = self.analysis_dir / 'reconstruction_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(reconstruction_stats, f, indent=2)
        
        return reconstruction_stats

    def create_summary_report(self, training_stats=None, reconstruction_stats=None):
        """创建总结报告"""
        print("📝 生成总结报告...")
        
        report_path = self.output_dir / 'experiment_summary.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# MAE 实验结果报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**设备**: {self.device}\n\n")
            
            # 训练统计
            if training_stats:
                f.write("## 训练统计\n\n")
                f.write(f"- 训练轮数: {training_stats['total_epochs']}\n")
                f.write(f"- 初始损失: {training_stats['initial_loss']:.4f}\n")
                f.write(f"- 最终损失: {training_stats['final_loss']:.4f}\n")
                f.write(f"- 损失下降: {training_stats['loss_reduction']:.1f}%\n")
                f.write(f"- 最高学习率: {training_stats['max_lr']:.2e}\n\n")
            
            # 重建统计
            if reconstruction_stats:
                f.write("## 重建性能\n\n")
                f.write("| 掩码比例 | 重建损失 | 平均误差 | 最大误差 |\n")
                f.write("|---------|---------|---------|----------|\n")
                for key, stats in reconstruction_stats.items():
                    mask_ratio = stats['mask_ratio']
                    f.write(f"| {mask_ratio*100:.0f}% | {stats['loss']:.4f} | {stats['mean_error']:.4f} | {stats['max_error']:.4f} |\n")
                f.write("\n")
            
            # 文件列表
            f.write("## 生成的文件\n\n")
            f.write("### 训练曲线\n")
            f.write("- `training_curves/training_curves.png`: 损失和学习率曲线\n\n")
            
            f.write("### 数据集样本\n")
            f.write("- `dataset_samples/dataset_samples_grid.png`: 数据集样本网格\n")
            f.write("- `dataset_samples/*_sample.png`: 各类别样本\n\n")
            
            f.write("### 重建结果\n")
            f.write("- `reconstructions/mae_reconstruction_comparison.png`: 不同掩码比例的重建对比\n\n")
            
            f.write("### 分析数据\n")
            f.write("- `analysis/training_stats.json`: 训练统计数据\n")
            f.write("- `analysis/reconstruction_stats.json`: 重建统计数据\n\n")
        
        print(f"✅ 实验报告保存: {report_path}")

    def run_complete_analysis(self):
        """运行完整分析"""
        print("🚀 开始完整的 MAE 可视化分析...")
        print("=" * 60)
        
        # 1. 分析训练日志
        training_stats = self.analyze_training_log()
        
        # 2. 展示数据集样本
        self.show_dataset_samples()
        
        # 3. MAE 重建可视化
        reconstruction_stats = self.visualize_mae_reconstruction()
        
        # 4. 生成总结报告
        self.create_summary_report(training_stats, reconstruction_stats)
        
        print("\n🎉 完整分析完成！")
        print(f"📁 所有结果保存在: {self.output_dir}")
        
        # 显示文件结构
        self.show_file_structure()
        
        return self.output_dir

    def show_file_structure(self):
        """显示生成的文件结构"""
        print(f"\n📁 生成的文件结构:")
        
        def print_tree(directory, prefix=""):
            items = sorted(directory.iterdir())
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                print(f"{prefix}{current_prefix}{item.name}")
                
                if item.is_dir() and not item.name.startswith('.'):
                    extension = "    " if is_last else "│   "
                    print_tree(item, prefix + extension)
        
        print_tree(self.output_dir)

def main():
    """主函数"""
    print("🎨 有组织的 MAE 可视化工具")
    print("=" * 50)
    
    # 创建可视化器
    visualizer = MAEVisualizer()
    
    # 运行完整分析
    output_dir = visualizer.run_complete_analysis()
    
    print(f"\n💡 提示:")
    print(f"  - 查看实验报告: open {output_dir}/experiment_summary.md")
    print(f"  - 浏览所有图片: open {output_dir}")
    print(f"  - 对比不同实验: ls {visualizer.base_output_dir}")

if __name__ == "__main__":
    main()


