#!/usr/bin/env python3
"""
解释PSNR概念和图像缩放效果
展示1920×1080缩放到224×224的实际效果
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

# 解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def explain_psnr():
    """详细解释PSNR概念"""
    
    print("📊 PSNR (Peak Signal-to-Noise Ratio) 详解")
    print("=" * 60)
    
    print("\n🎯 PSNR是什么？")
    print("PSNR是衡量图像质量的重要指标，用于评估重建/压缩图像与原图的相似度")
    
    print("\n📐 PSNR计算公式：")
    print("PSNR = 20 × log10(MAX_PIXEL_VALUE / √MSE)")
    print("其中：")
    print("  • MAX_PIXEL_VALUE = 1.0 (归一化后的最大像素值)")
    print("  • MSE = Mean Squared Error (均方误差)")
    print("  • MSE = mean((original - reconstructed)²)")
    
    print("\n📈 PSNR值的含义：")
    quality_levels = [
        ("🟢 优秀", "> 30dB", "几乎看不出差异，专业级质量"),
        ("🔵 很好", "25-30dB", "轻微差异，高质量重建"),
        ("🟡 良好", "20-25dB", "可接受的质量，有明显但不严重的差异"),
        ("🟠 一般", "15-20dB", "质量下降明显，但仍可用"),
        ("🔴 较差", "10-15dB", "质量很差，有严重的失真"),
        ("⚫ 很差", "< 10dB", "几乎无法使用，严重失真")
    ]
    
    for level, range_str, description in quality_levels:
        print(f"  {level} {range_str:>8}: {description}")
    
    print(f"\n💡 我们的实验结果：")
    print(f"  • 当前PSNR: 9-12dB → 🔴 质量较差")
    print(f"  • 目标PSNR: >20dB → 🟡 可接受质量")
    print(f"  • 理想PSNR: >25dB → 🔵 高质量重建")

def demonstrate_psnr_calculation():
    """演示PSNR计算过程"""
    print(f"\n🧮 PSNR计算演示...")
    
    # 创建示例图像
    original = torch.rand(3, 100, 100)  # 原始图像
    
    # 创建不同质量的"重建"图像
    reconstructions = {
        '完美重建': original.clone(),
        '高质量重建': original + torch.randn_like(original) * 0.05,
        '中等质量重建': original + torch.randn_like(original) * 0.1,
        '低质量重建': original + torch.randn_like(original) * 0.2,
        '很差重建': torch.rand_like(original)  # 随机噪声
    }
    
    print(f"📊 不同重建质量的PSNR值：")
    
    fig, axes = plt.subplots(1, len(reconstructions), figsize=(len(reconstructions)*3, 3))
    
    for i, (name, recon) in enumerate(reconstructions.items()):
        # 计算MSE
        mse = torch.mean((original - recon)**2).item()
        
        # 计算PSNR
        if mse > 0:
            psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(torch.tensor(mse))).item()
        else:
            psnr = float('inf')
        
        # 显示图像
        axes[i].imshow(torch.clamp(recon, 0, 1).permute(1, 2, 0))
        axes[i].set_title(f'{name}\nPSNR: {psnr:.1f}dB')
        axes[i].axis('off')
        
        print(f"  {name}: MSE={mse:.6f}, PSNR={psnr:.1f}dB")
    
    plt.tight_layout()
    plt.savefig('psnr_demonstration.png', dpi=150, bbox_inches='tight')
    print("✅ PSNR演示保存: psnr_demonstration.png")
    plt.close()

def show_scaling_effects():
    """展示图像缩放效果"""
    print(f"\n🖼️ 展示1920×1080缩放到224×224的效果...")
    
    try:
        # 加载AnimeDiffusion数据集
        ds = load_dataset("Mercity/AnimeDiffusion_Dataset")
        
        # 选择几张不同风格的图片
        sample_indices = [0, 100, 500, 1000, 2000]
        
        fig, axes = plt.subplots(len(sample_indices), 4, figsize=(16, len(sample_indices)*3))
        
        scaling_methods = [
            ('原图 1920×1080', None),
            ('直接缩放 224×224', transforms.Resize((224, 224))),
            ('智能裁剪 224×224', transforms.Compose([
                transforms.Resize(int(224 * 1.15)),
                transforms.CenterCrop(224)
            ])),
            ('保持比例缩放', transforms.Compose([
                transforms.Resize(224),  # 保持宽高比
                transforms.CenterCrop(224)
            ]))
        ]
        
        for i, idx in enumerate(sample_indices):
            try:
                sample = ds['train'][idx]
                original_img = sample['image']
                
                if original_img.mode != 'RGB':
                    original_img = original_img.convert('RGB')
                
                print(f"\n图片 {i+1} (索引 {idx}):")
                print(f"  原始尺寸: {original_img.size}")
                
                for j, (method_name, transform_method) in enumerate(scaling_methods):
                    if transform_method is None:
                        # 显示原图的缩略图
                        display_img = original_img.resize((224, 126))  # 保持16:9比例用于显示
                        axes[i, j].imshow(display_img)
                        
                        # 计算信息损失
                        original_pixels = 1920 * 1080
                        target_pixels = 224 * 224
                        info_loss = (1 - target_pixels / original_pixels) * 100
                        
                        axes[i, j].set_title(f'{method_name}\n信息保留: {100-info_loss:.1f}%')
                    else:
                        # 应用缩放变换
                        scaled_img = transform_method(original_img)
                        axes[i, j].imshow(scaled_img)
                        axes[i, j].set_title(f'{method_name}\n{scaled_img.size}')
                    
                    axes[i, j].axis('off')
                
                # 计算缩放损失
                original_area = 1920 * 1080
                target_area = 224 * 224
                scale_factor = target_area / original_area
                print(f"  缩放比例: {scale_factor:.6f} ({scale_factor*100:.3f}%)")
                print(f"  信息损失: {(1-scale_factor)*100:.1f}%")
                
            except Exception as e:
                print(f"处理图片 {idx} 时出错: {e}")
                for j in range(4):
                    axes[i, j].text(0.5, 0.5, f'Error', ha='center', va='center')
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig('image_scaling_effects.png', dpi=150, bbox_inches='tight')
        print("✅ 图像缩放效果保存: image_scaling_effects.png")
        plt.close()
        
    except Exception as e:
        print(f"缩放效果演示失败: {e}")

def analyze_information_loss():
    """分析信息损失"""
    print(f"\n📉 分析缩放造成的信息损失...")
    
    # 计算不同缩放的信息损失
    resolutions = [
        ('原始', 1920, 1080),
        ('高清', 1280, 720),
        ('标清', 640, 480),
        ('MAE输入', 224, 224),
        ('缩略图', 112, 112)
    ]
    
    original_pixels = 1920 * 1080
    
    print(f"📊 不同分辨率的信息保留率:")
    print(f"{'分辨率':<12} {'像素数':<12} {'信息保留':<12} {'损失程度'}")
    print("-" * 50)
    
    info_retention = []
    labels = []
    
    for name, w, h in resolutions:
        pixels = w * h
        retention = pixels / original_pixels * 100
        loss = 100 - retention
        
        if retention > 50:
            loss_level = "轻微"
        elif retention > 10:
            loss_level = "中等"
        elif retention > 1:
            loss_level = "严重"
        else:
            loss_level = "极严重"
        
        print(f"{name:<12} {pixels:<12,} {retention:<11.2f}% {loss_level}")
        
        info_retention.append(retention)
        labels.append(f"{name}\n{w}×{h}")
    
    # 可视化信息损失
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 信息保留率柱状图
    bars = ax1.bar(labels, info_retention, color=['green', 'blue', 'orange', 'red', 'darkred'], alpha=0.7)
    ax1.set_ylabel('Information Retention (%)')
    ax1.set_title('Information Retention by Resolution')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, retention in zip(bars, info_retention):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{retention:.1f}%', ha='center', va='bottom')
    
    # 像素数对比
    pixel_counts = [w*h for _, w, h in resolutions]
    ax2.bar(labels, pixel_counts, color=['green', 'blue', 'orange', 'red', 'darkred'], alpha=0.7)
    ax2.set_ylabel('Total Pixels')
    ax2.set_title('Pixel Count by Resolution')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_yscale('log')  # 使用对数刻度
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('information_loss_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ 信息损失分析保存: information_loss_analysis.png")
    plt.close()
    
    print(f"\n💡 关键发现:")
    print(f"  • 1920×1080 → 224×224: 损失 {100-info_retention[3]:.1f}% 的信息！")
    print(f"  • 这相当于丢弃了 {100-info_retention[3]:.0f}% 的像素细节")
    print(f"  • 这可能是重建质量差的重要原因")

def show_actual_scaled_images():
    """展示实际缩放后的图像质量"""
    print(f"\n🎨 展示实际缩放后的图像...")
    
    try:
        ds = load_dataset("Mercity/AnimeDiffusion_Dataset")
        
        # 选择几张代表性图片
        test_indices = [0, 100, 500]
        
        fig, axes = plt.subplots(len(test_indices), 3, figsize=(12, len(test_indices)*4))
        
        for i, idx in enumerate(test_indices):
            sample = ds['train'][idx]
            original_img = sample['image']
            prompt = sample.get('long_prompt', sample.get('short_prompt', ''))[:60] + "..."
            
            if original_img.mode != 'RGB':
                original_img = original_img.convert('RGB')
            
            print(f"\n图片 {i+1}:")
            print(f"  原始尺寸: {original_img.size}")
            print(f"  描述: {prompt}")
            
            # 原图 (缩放用于显示)
            display_original = original_img.resize((400, 225))  # 保持16:9比例
            axes[i, 0].imshow(display_original)
            axes[i, 0].set_title(f'Original HD\n1920×1080\n({prompt[:30]}...)')
            axes[i, 0].axis('off')
            
            # 直接缩放到224×224
            scaled_direct = original_img.resize((224, 224))
            axes[i, 1].imshow(scaled_direct)
            axes[i, 1].set_title('Direct Resize\n224×224\n(May be distorted)')
            axes[i, 1].axis('off')
            
            # 智能裁剪到224×224 (我们实际使用的方法)
            smart_crop_transform = transforms.Compose([
                transforms.Resize(int(224 * 1.15)),
                transforms.CenterCrop(224)
            ])
            scaled_smart = smart_crop_transform(original_img)
            axes[i, 2].imshow(scaled_smart)
            axes[i, 2].set_title('Smart Crop\n224×224\n(Our method)')
            axes[i, 2].axis('off')
            
            # 计算质量损失
            original_array = np.array(original_img.resize((224, 224)))
            smart_array = np.array(scaled_smart)
            
            mse = np.mean((original_array.astype(float) - smart_array.astype(float))**2) / (255**2)
            if mse > 0:
                psnr = 20 * np.log10(1.0 / np.sqrt(mse))
                print(f"  缩放质量损失: PSNR {psnr:.1f}dB")
            else:
                print(f"  缩放质量损失: 无损失")
        
        plt.tight_layout()
        plt.savefig('actual_scaling_comparison.png', dpi=150, bbox_inches='tight')
        print("✅ 实际缩放对比保存: actual_scaling_comparison.png")
        plt.close()
        
    except Exception as e:
        print(f"缩放演示失败: {e}")

def create_psnr_quality_examples():
    """创建不同PSNR质量的示例"""
    print(f"\n🎭 创建不同PSNR质量的示例...")
    
    # 创建一个清晰的测试图像
    test_img = torch.zeros(3, 224, 224)
    
    # 添加一些图案
    # 渐变背景
    for i in range(224):
        test_img[:, i, :] = i / 224
    
    # 添加一些几何图形
    # 白色圆形
    center = 112
    radius = 40
    y, x = torch.meshgrid(torch.arange(224), torch.arange(224), indexing='ij')
    circle_mask = (x - center)**2 + (y - center)**2 <= radius**2
    test_img[:, circle_mask] = 1.0
    
    # 黑色矩形
    test_img[:, 50:100, 150:200] = 0.0
    
    # 创建不同质量的版本
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    quality_names = ['Perfect', 'Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (noise_level, quality_name) in enumerate(zip(noise_levels, quality_names)):
        # 添加噪声
        noisy_img = test_img + torch.randn_like(test_img) * noise_level
        noisy_img = torch.clamp(noisy_img, 0, 1)
        
        # 计算PSNR
        mse = torch.mean((test_img - noisy_img)**2).item()
        if mse > 0:
            psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(torch.tensor(mse))).item()
        else:
            psnr = float('inf')
        
        # 显示
        axes[i].imshow(noisy_img.permute(1, 2, 0))
        axes[i].set_title(f'{quality_name}\nNoise: {noise_level:.2f}\nPSNR: {psnr:.1f}dB')
        axes[i].axis('off')
        
        print(f"  {quality_name}: 噪声{noise_level:.2f}, PSNR {psnr:.1f}dB")
    
    plt.tight_layout()
    plt.savefig('psnr_quality_examples.png', dpi=150, bbox_inches='tight')
    print("✅ PSNR质量示例保存: psnr_quality_examples.png")
    plt.close()

def analyze_our_results():
    """分析我们的实验结果"""
    print(f"\n📊 分析我们的MAE实验结果...")
    
    experiments = [
        ('合成数据集', '几何图案', '224×224', '1.31', '约8dB'),
        ('anime-captions', '动漫图片', '512×512→224×224', '1.07', '约9dB'),
        ('AnimeDiffusion-75%', '高质量动漫', '1920×1080→224×224', '0.95', '约10dB'),
        ('AnimeDiffusion-25%', '高质量动漫', '1920×1080→224×224', '0.81', '约11dB'),
        ('改进版25%', '高质量动漫', '1920×1080→224×224', '0.73', '约12dB')
    ]
    
    print(f"🎯 我们的实验PSNR分析:")
    print(f"{'实验':<20} {'数据类型':<12} {'分辨率':<20} {'损失':<8} {'估计PSNR'}")
    print("-" * 80)
    
    for exp_name, data_type, resolution, loss, psnr in experiments:
        print(f"{exp_name:<20} {data_type:<12} {resolution:<20} {loss:<8} {psnr}")
    
    print(f"\n💡 关键观察:")
    print(f"  • 所有实验的PSNR都在8-12dB范围")
    print(f"  • 这个范围属于 🔴 质量较差 级别")
    print(f"  • 主要原因：")
    print(f"    1. 巨大的分辨率损失 (97.6%信息丢失)")
    print(f"    2. MAE架构限制")
    print(f"    3. 训练时间不足")
    
    print(f"\n🎯 要达到可接受质量 (PSNR > 20dB):")
    print(f"  • 需要将MSE从当前的~0.1降到<0.01")
    print(f"  • 这需要重建误差减少10倍以上")
    print(f"  • 可能需要根本性的方法改变")

def main():
    """主函数"""
    print("📊 PSNR和图像缩放详解")
    print("=" * 50)
    
    # 1. 解释PSNR概念
    explain_psnr()
    
    # 2. 演示PSNR计算
    demonstrate_psnr_calculation()
    
    # 3. 展示缩放效果
    show_scaling_effects()
    
    # 4. 创建质量示例
    create_psnr_quality_examples()
    
    # 5. 分析我们的结果
    analyze_our_results()
    
    print(f"\n🎉 总结:")
    print(f"  📊 PSNR: 图像质量的客观指标，>20dB才算可接受")
    print(f"  🖼️  缩放: 1920×1080→224×224损失了97.6%的信息")
    print(f"  🔍 我们的结果: 9-12dB，属于质量较差级别")
    print(f"  💡 改进方向: 需要大幅提升重建精度")

if __name__ == "__main__":
    main()


