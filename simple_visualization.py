#!/usr/bin/env python3
"""
简化版 MAE 可视化工具
展示训练结果和模型重建效果
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json

# 解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import models_mae

def visualize_mae_with_random_model():
    """使用随机初始化的模型展示 MAE 工作原理"""
    print("🎨 使用随机模型展示 MAE 重建过程")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 创建模型
    model = models_mae.mae_vit_base_patch16(norm_pix_loss=True)
    model.to(device)
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
    
    # 找一张测试图片
    test_img_path = './test_dataset/train/class_00/img_0000.png'
    if not os.path.exists(test_img_path):
        print(f"❌ 测试图片不存在: {test_img_path}")
        return
    
    # 加载图片
    original_img = Image.open(test_img_path).convert('RGB')
    img_tensor = transform(original_img).unsqueeze(0).to(device)
    
    # 测试不同的掩码比例
    mask_ratios = [0.5, 0.75, 0.9]
    
    fig, axes = plt.subplots(len(mask_ratios), 3, figsize=(12, len(mask_ratios)*4))
    
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
        
        # 显示结果
        if len(mask_ratios) == 1:
            ax_row = axes
        else:
            ax_row = axes[i]
            
        ax_row[0].imshow(original_display.permute(1, 2, 0))
        ax_row[0].set_title('Original Image')
        ax_row[0].axis('off')
        
        ax_row[1].imshow(masked_img.permute(1, 2, 0))
        ax_row[1].set_title(f'Masked ({mask_ratio*100:.0f}% hidden)')
        ax_row[1].axis('off')
        
        ax_row[2].imshow(reconstructed_display.permute(1, 2, 0))
        ax_row[2].set_title(f'Reconstructed\nLoss: {loss.item():.3f}')
        ax_row[2].axis('off')
        
        print(f"  掩码比例 {mask_ratio*100:.0f}%: 损失 {loss.item():.4f}")
    
    plt.tight_layout()
    plt.savefig('mae_reconstruction_demo.png', dpi=150, bbox_inches='tight')
    print("✅ MAE 重建演示保存: mae_reconstruction_demo.png")
    plt.show()

def analyze_training_log():
    """分析训练日志"""
    print("📊 分析训练日志")
    
    log_file = './output_m4/log.txt'
    if not os.path.exists(log_file):
        print(f"❌ 日志文件不存在: {log_file}")
        return
    
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
        return
    
    print(f"📈 训练统计:")
    print(f"  训练轮数: {len(epochs)}")
    print(f"  初始损失: {losses[0]:.4f}")
    print(f"  最终损失: {losses[-1]:.4f}")
    print(f"  损失下降: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    print(f"  最高学习率: {max(lrs):.2e}")
    
    return epochs, losses, lrs

def show_test_dataset_samples():
    """展示测试数据集样本"""
    print("🖼️  展示测试数据集样本")
    
    test_dir = './test_dataset/train'
    if not os.path.exists(test_dir):
        print(f"❌ 测试数据集不存在: {test_dir}")
        return
    
    # 收集每个类别的第一张图片
    class_samples = []
    class_names = sorted(os.listdir(test_dir))
    
    for class_name in class_names[:5]:  # 最多5个类别
        class_path = os.path.join(test_dir, class_name)
        if os.path.isdir(class_path):
            img_files = sorted(os.listdir(class_path))
            if img_files:
                img_path = os.path.join(class_path, img_files[0])
                class_samples.append((class_name, img_path))
    
    if not class_samples:
        print("❌ 未找到测试图片")
        return
    
    # 创建可视化
    n_samples = len(class_samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples*3, 3))
    
    if n_samples == 1:
        axes = [axes]
    
    for i, (class_name, img_path) in enumerate(class_samples):
        try:
            img = Image.open(img_path).convert('RGB')
            axes[i].imshow(img)
            axes[i].set_title(f'{class_name}')
            axes[i].axis('off')
        except Exception as e:
            print(f"加载图片失败 {img_path}: {e}")
    
    plt.tight_layout()
    plt.savefig('test_dataset_samples.png', dpi=150, bbox_inches='tight')
    print("✅ 数据集样本保存: test_dataset_samples.png")
    plt.show()

def main():
    """主函数"""
    print("🎨 MAE 简化可视化工具")
    print("=" * 50)
    
    # 1. 分析训练日志
    training_stats = analyze_training_log()
    
    # 2. 展示数据集样本
    show_test_dataset_samples()
    
    # 3. MAE 重建演示
    visualize_mae_with_random_model()
    
    print("\n🎉 可视化完成！")
    print("生成的文件:")
    print("  - training_curves.png: 训练曲线")
    print("  - test_dataset_samples.png: 数据集样本")
    print("  - mae_reconstruction_demo.png: MAE 重建演示")
    
    # 4. 显示文件列表
    print("\n📁 当前目录的可视化文件:")
    import glob
    png_files = glob.glob("*.png")
    for png_file in sorted(png_files):
        size = os.path.getsize(png_file) / 1024  # KB
        print(f"  {png_file}: {size:.1f} KB")

if __name__ == "__main__":
    main()


