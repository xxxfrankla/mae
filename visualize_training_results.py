#!/usr/bin/env python3
"""
MAE 训练结果可视化工具
展示训练过程、模型输出和重建效果
"""

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# 解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import models_mae

def load_trained_model(checkpoint_path, device='mps'):
    """加载训练好的模型"""
    print(f"🔄 加载模型: {checkpoint_path}")
    
    model = models_mae.mae_vit_base_patch16(norm_pix_loss=True)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print(f"✅ 模型加载成功，epoch: {checkpoint.get('epoch', 'unknown')}")
    else:
        print("⚠️  未找到checkpoint，使用随机初始化的模型")
    
    model.to(device)
    model.eval()
    return model

def visualize_mae_reconstruction(model, image_path, device='mps', mask_ratio=0.75):
    """可视化 MAE 重建过程"""
    print(f"🎨 可视化重建过程: {image_path}")
    
    # 加载和预处理图像
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 反归一化用于显示
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # 加载图像
    original_img = Image.open(image_path).convert('RGB')
    img_tensor = transform(original_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 前向传播
        loss, pred, mask = model(img_tensor, mask_ratio=mask_ratio)
        
        # 重建图像
        reconstructed = model.unpatchify(pred)
        
        # 创建掩码可视化
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
        mask = model.unpatchify(mask)
    
    # 转换为可显示格式
    original_display = inv_normalize(img_tensor[0]).cpu()
    original_display = torch.clamp(original_display, 0, 1)
    
    reconstructed_display = inv_normalize(reconstructed[0]).cpu()
    reconstructed_display = torch.clamp(reconstructed_display, 0, 1)
    
    mask_display = mask[0].cpu()
    
    # 创建掩码图像（被掩盖的区域显示为灰色）
    masked_img = original_display.clone()
    masked_img = masked_img * (1 - mask_display) + mask_display * 0.5
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 原始图像
    axes[0, 0].imshow(original_display.permute(1, 2, 0))
    axes[0, 0].set_title('原始图像', fontsize=14)
    axes[0, 0].axis('off')
    
    # 掩码图像
    axes[0, 1].imshow(masked_img.permute(1, 2, 0))
    axes[0, 1].set_title(f'掩码图像 ({mask_ratio*100:.0f}% 被掩盖)', fontsize=14)
    axes[0, 1].axis('off')
    
    # 重建图像
    axes[1, 0].imshow(reconstructed_display.permute(1, 2, 0))
    axes[1, 0].set_title('重建图像', fontsize=14)
    axes[1, 0].axis('off')
    
    # 重建误差
    error = torch.abs(original_display - reconstructed_display)
    error_display = error.mean(dim=0)  # 平均RGB通道
    im = axes[1, 1].imshow(error_display, cmap='hot')
    axes[1, 1].set_title('重建误差 (越亮误差越大)', fontsize=14)
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    
    # 保存结果
    output_path = f'mae_reconstruction_{Path(image_path).stem}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化结果保存: {output_path}")
    
    # 显示统计信息
    print(f"📊 重建统计:")
    print(f"  重建损失: {loss.item():.4f}")
    print(f"  掩码比例: {mask.float().mean().item():.2%}")
    print(f"  平均重建误差: {error.mean().item():.4f}")
    
    plt.show()
    return loss.item(), mask.float().mean().item()

def plot_training_curves(log_file='./output_m4/log.txt'):
    """绘制训练曲线"""
    print(f"📈 绘制训练曲线: {log_file}")
    
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
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 损失曲线
    ax1.plot(epochs, losses, 'b-', linewidth=2, label='训练损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('MAE 训练损失曲线')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 学习率曲线
    ax2.plot(epochs, lrs, 'r-', linewidth=2, label='学习率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('学习率变化曲线')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("✅ 训练曲线保存: training_curves.png")
    plt.show()

def analyze_model_outputs(model, test_images_dir='./test_dataset/train', device='mps'):
    """分析模型在不同类别上的表现"""
    print(f"🔍 分析模型输出: {test_images_dir}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    class_losses = {}
    
    # 遍历每个类别
    for class_dir in sorted(os.listdir(test_images_dir)):
        class_path = os.path.join(test_images_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        
        losses = []
        
        # 测试该类别的前5张图片
        image_files = sorted(os.listdir(class_path))[:5]
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    loss, _, _ = model(img_tensor, mask_ratio=0.75)
                    losses.append(loss.item())
            except:
                continue
        
        if losses:
            class_losses[class_dir] = np.mean(losses)
    
    # 可视化类别损失
    if class_losses:
        classes = list(class_losses.keys())
        losses = list(class_losses.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes, losses, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.xlabel('类别')
        plt.ylabel('平均重建损失')
        plt.title('不同类别的重建损失对比')
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar, loss in zip(bars, losses):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{loss:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('class_analysis.png', dpi=150, bbox_inches='tight')
        print("✅ 类别分析保存: class_analysis.png")
        plt.show()
        
        # 打印统计信息
        print(f"📊 类别分析结果:")
        for class_name, loss in sorted(class_losses.items(), key=lambda x: x[1]):
            print(f"  {class_name}: {loss:.4f}")

def create_reconstruction_grid(model, test_images_dir='./test_dataset/train', device='mps'):
    """创建重建结果网格"""
    print(f"🎨 创建重建网格")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # 收集测试图片
    test_images = []
    class_dirs = sorted(os.listdir(test_images_dir))[:3]  # 前3个类别
    
    for class_dir in class_dirs:
        class_path = os.path.join(test_images_dir, class_dir)
        if os.path.isdir(class_path):
            img_files = sorted(os.listdir(class_path))[:2]  # 每类2张
            for img_file in img_files:
                test_images.append(os.path.join(class_path, img_file))
    
    # 创建网格
    n_images = min(6, len(test_images))
    fig, axes = plt.subplots(3, n_images, figsize=(n_images*3, 9))
    
    for i, img_path in enumerate(test_images[:n_images]):
        try:
            # 加载图像
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                loss, pred, mask = model(img_tensor, mask_ratio=0.75)
                reconstructed = model.unpatchify(pred)
                
                # 创建掩码
                mask_vis = mask.detach()
                mask_vis = mask_vis.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
                mask_vis = model.unpatchify(mask_vis)
            
            # 转换显示格式
            original = torch.clamp(inv_normalize(img_tensor[0]).cpu(), 0, 1)
            recon = torch.clamp(inv_normalize(reconstructed[0]).cpu(), 0, 1)
            masked = original * (1 - mask_vis[0].cpu()) + mask_vis[0].cpu() * 0.5
            
            # 显示
            axes[0, i].imshow(original.permute(1, 2, 0))
            axes[0, i].set_title(f'原图 {i+1}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(masked.permute(1, 2, 0))
            axes[1, i].set_title(f'掩码 {i+1}')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(recon.permute(1, 2, 0))
            axes[2, i].set_title(f'重建 {i+1}\nLoss: {loss.item():.3f}')
            axes[2, i].axis('off')
            
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
    
    plt.tight_layout()
    plt.savefig('reconstruction_grid.png', dpi=150, bbox_inches='tight')
    print("✅ 重建网格保存: reconstruction_grid.png")
    plt.show()

def main():
    """主函数"""
    print("🎨 MAE 训练结果可视化工具")
    print("=" * 50)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 绘制训练曲线
    plot_training_curves()
    
    # 2. 加载模型
    checkpoint_path = './output_m4/checkpoint-1.pth'  # 最后一个epoch的checkpoint
    model = load_trained_model(checkpoint_path, device)
    
    # 3. 单张图片重建可视化
    test_img = './test_dataset/train/class_00/img_0000.png'
    if os.path.exists(test_img):
        visualize_mae_reconstruction(model, test_img, device)
    
    # 4. 类别分析
    analyze_model_outputs(model, device=device)
    
    # 5. 重建网格
    create_reconstruction_grid(model, device=device)
    
    print("\n🎉 可视化完成！")
    print("生成的文件:")
    print("  - training_curves.png: 训练曲线")
    print("  - mae_reconstruction_*.png: 单张图片重建")
    print("  - class_analysis.png: 类别分析")
    print("  - reconstruction_grid.png: 重建网格")

if __name__ == "__main__":
    main()


