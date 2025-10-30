#!/usr/bin/env python3
"""
调试MAE重建质量问题
分析为什么重建图像模糊，并提供解决方案
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

import models_mae

def analyze_reconstruction_issues():
    """分析重建质量问题"""
    print("🔍 分析MAE重建质量问题...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 加载不同的模型进行对比
    models_to_test = [
        {
            'name': '25%掩码训练模型',
            'path': './output_animediffusion_mask25/checkpoint-9.pth',
            'mask_ratio': 0.25
        },
        {
            'name': '75%掩码训练模型', 
            'path': './output_animediffusion/checkpoint-4.pth',
            'mask_ratio': 0.75
        },
        {
            'name': '随机初始化模型',
            'path': None,
            'mask_ratio': 0.25
        }
    ]
    
    # 加载测试图片
    try:
        ds = load_dataset("Mercity/AnimeDiffusion_Dataset")
        test_sample = ds['train'][0]
        original_img = test_sample['image']
        
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        
        print(f"✅ 测试图片加载成功: {original_img.size}")
    except Exception as e:
        print(f"❌ 测试图片加载失败: {e}")
        return
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(int(224 * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    img_tensor = transform(original_img).unsqueeze(0).to(device)
    original_display = torch.clamp(inv_normalize(img_tensor[0]).cpu(), 0, 1)
    
    fig, axes = plt.subplots(len(models_to_test), 4, figsize=(16, len(models_to_test)*3))
    
    for i, model_info in enumerate(models_to_test):
        print(f"\n🤖 测试 {model_info['name']}...")
        
        # 加载模型
        model = models_mae.mae_vit_base_patch16(norm_pix_loss=True)
        
        if model_info['path'] and os.path.exists(model_info['path']):
            try:
                checkpoint = torch.load(model_info['path'], map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['model'])
                print(f"  ✅ 模型加载成功")
            except Exception as e:
                print(f"  ⚠️  模型加载失败: {e}")
        else:
            print(f"  ⚠️  使用随机初始化模型")
        
        model.to(device)
        model.eval()
        
        # 进行重建
        with torch.no_grad():
            loss, pred, mask = model(img_tensor, mask_ratio=model_info['mask_ratio'])
            reconstructed = model.unpatchify(pred)
            
            # 创建掩码可视化
            mask_vis = mask.detach().unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
            mask_vis = model.unpatchify(mask_vis)
        
        # 显示结果
        masked_img = original_display * (1 - mask_vis[0].cpu()) + mask_vis[0].cpu() * 0.5
        recon_display = torch.clamp(inv_normalize(reconstructed[0]).cpu(), 0, 1)
        
        # 计算重建质量指标
        mse = torch.mean((original_display - recon_display)**2).item()
        psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.mean((original_display - recon_display)**2))).item()
        
        # 显示
        axes[i, 0].imshow(original_display.permute(1, 2, 0))
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(masked_img.permute(1, 2, 0))
        axes[i, 1].set_title(f'{model_info["mask_ratio"]*100:.0f}% Masked')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(recon_display.permute(1, 2, 0))
        axes[i, 2].set_title(f'{model_info["name"]}\nLoss: {loss.item():.3f}\nPSNR: {psnr:.1f}dB')
        axes[i, 2].axis('off')
        
        # 显示误差图
        error = torch.abs(original_display - recon_display).mean(dim=0)
        im = axes[i, 3].imshow(error, cmap='hot', vmin=0, vmax=0.3)
        axes[i, 3].set_title(f'Error Map\nMSE: {mse:.4f}')
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
        
        print(f"  损失: {loss.item():.4f}, PSNR: {psnr:.1f}dB, MSE: {mse:.4f}")
    
    plt.tight_layout()
    plt.savefig('reconstruction_quality_debug.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 重建质量调试图保存: reconstruction_quality_debug.png")
    plt.close()

def identify_blur_causes():
    """识别模糊的原因"""
    print(f"\n🔍 识别重建模糊的可能原因:")
    
    causes_and_solutions = [
        {
            'cause': '训练不充分',
            'description': '模型还没有学会有效的重建',
            'solution': '增加训练轮数到50-100个epoch',
            'priority': 'HIGH'
        },
        {
            'cause': '学习率过高',
            'description': '优化步长太大，导致不稳定',
            'solution': '降低学习率 (blr=1e-4 或更低)',
            'priority': 'MEDIUM'
        },
        {
            'cause': '掩码比例不当',
            'description': '25%可能太简单，75%可能太难',
            'solution': '尝试50%掩码比例',
            'priority': 'MEDIUM'
        },
        {
            'cause': '数据预处理问题',
            'description': '图像缩放或归一化不当',
            'solution': '检查图像预处理流程',
            'priority': 'HIGH'
        },
        {
            'cause': '模型容量不足',
            'description': 'ViT-Base可能对高质量图像不够',
            'solution': '尝试ViT-Large模型',
            'priority': 'LOW'
        }
    ]
    
    for i, item in enumerate(causes_and_solutions, 1):
        priority_color = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}
        print(f"\n{i}. {priority_color[item['priority']]} {item['cause']}")
        print(f"   原因: {item['description']}")
        print(f"   解决方案: {item['solution']}")

def create_optimized_training_configs():
    """创建优化的训练配置"""
    print(f"\n🎯 创建优化的25%掩码训练配置...")
    
    configs = {
        'improved_25_mask': {
            'mask_ratio': 0.25,
            'epochs': 50,
            'batch_size': 6,
            'blr': 1e-4,  # 降低学习率
            'warmup_epochs': 10,
            'max_samples': 1000,
            'description': '改进的25%掩码配置'
        },
        'balanced_50_mask': {
            'mask_ratio': 0.5,
            'epochs': 30,
            'batch_size': 6,
            'blr': 1.2e-4,
            'warmup_epochs': 8,
            'max_samples': 1000,
            'description': '平衡的50%掩码配置'
        },
        'fine_tuned_25': {
            'mask_ratio': 0.25,
            'epochs': 100,
            'batch_size': 4,
            'blr': 8e-5,  # 更低的学习率
            'warmup_epochs': 15,
            'max_samples': 2000,
            'description': '精细调优的25%掩码配置'
        }
    }
    
    print(f"\n📋 推荐的优化配置:")
    print(f"{'配置名':<20} {'掩码':<6} {'轮数':<6} {'批次':<6} {'学习率':<10} {'描述'}")
    print("-" * 80)
    
    for name, config in configs.items():
        print(f"{name:<20} {config['mask_ratio']*100:.0f}%{'':<3} {config['epochs']:<6} {config['batch_size']:<6} {config['blr']:.1e}{'':<3} {config['description']}")
    
    # 生成训练命令
    print(f"\n🚀 推荐的训练命令:")
    
    for name, config in configs.items():
        print(f"\n# {config['description']}")
        cmd = f"python main_pretrain_animediffusion.py \\\n"
        cmd += f"    --mask_ratio {config['mask_ratio']} \\\n"
        cmd += f"    --epochs {config['epochs']} \\\n"
        cmd += f"    --batch_size {config['batch_size']} \\\n"
        cmd += f"    --blr {config['blr']:.1e} \\\n"
        cmd += f"    --warmup_epochs {config['warmup_epochs']} \\\n"
        cmd += f"    --max_samples {config['max_samples']} \\\n"
        cmd += f"    --output_dir ./output_{name} \\\n"
        cmd += f"    --log_dir ./output_{name}"
        print(cmd)
    
    return configs

def test_different_normalizations():
    """测试不同的归一化方法"""
    print(f"\n🧪 测试不同的图像归一化方法...")
    
    try:
        ds = load_dataset("Mercity/AnimeDiffusion_Dataset")
        original_img = ds['train'][0]['image']
        
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        
        # 测试不同的归一化方法
        normalizations = [
            {
                'name': 'ImageNet标准',
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            {
                'name': '零均值单位方差',
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5]
            },
            {
                'name': '无归一化',
                'mean': [0.0, 0.0, 0.0],
                'std': [1.0, 1.0, 1.0]
            }
        ]
        
        fig, axes = plt.subplots(1, len(normalizations), figsize=(len(normalizations)*4, 4))
        
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model = models_mae.mae_vit_base_patch16(norm_pix_loss=True)
        model.to(device)
        model.eval()
        
        for i, norm_config in enumerate(normalizations):
            # 创建变换
            transform = transforms.Compose([
                transforms.Resize(int(224 * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_config['mean'], std=norm_config['std'])
            ])
            
            inv_normalize = transforms.Normalize(
                mean=[-m/s for m, s in zip(norm_config['mean'], norm_config['std'])],
                std=[1/s for s in norm_config['std']]
            )
            
            # 处理图像
            img_tensor = transform(original_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                loss, pred, mask = model(img_tensor, mask_ratio=0.25)
                reconstructed = model.unpatchify(pred)
            
            # 反归一化显示
            recon_display = torch.clamp(inv_normalize(reconstructed[0]).cpu(), 0, 1)
            
            axes[i].imshow(recon_display.permute(1, 2, 0))
            axes[i].set_title(f'{norm_config["name"]}\nLoss: {loss.item():.3f}')
            axes[i].axis('off')
            
            print(f"  {norm_config['name']}: 损失 {loss.item():.4f}")
        
        plt.tight_layout()
        plt.savefig('normalization_comparison.png', dpi=150, bbox_inches='tight')
        print(f"✅ 归一化对比保存: normalization_comparison.png")
        plt.close()
        
    except Exception as e:
        print(f"归一化测试失败: {e}")

def create_image_repair_config():
    """创建专门用于图像修复的配置"""
    print(f"\n🛠️ 创建图像修复专用配置...")
    
    repair_config = {
        # 基本参数
        'mask_ratio': 0.25,  # 较低的掩码比例，适合修复
        'epochs': 100,       # 充分训练
        'batch_size': 4,     # 较小批次，更稳定
        'accum_iter': 16,    # 保持有效批次大小
        
        # 学习率优化
        'blr': 5e-5,         # 更低的基础学习率
        'warmup_epochs': 20, # 更长的预热
        'min_lr': 1e-6,      # 设置最小学习率
        
        # 数据处理
        'input_size': 224,   # 标准尺寸
        'resize_strategy': 'smart_crop',
        'max_samples': 2000, # 适中的样本数
        
        # 正则化
        'weight_decay': 0.02, # 降低权重衰减
        'norm_pix_loss': True,
        
        # 保存策略
        'save_freq': 10,
        'output_dir': './output_image_repair',
        'log_dir': './output_image_repair'
    }
    
    print(f"📋 图像修复专用配置:")
    for key, value in repair_config.items():
        print(f"  {key}: {value}")
    
    # 生成训练命令
    cmd = "# 图像修复专用MAE训练\n"
    cmd += "python main_pretrain_animediffusion.py \\\n"
    for key, value in repair_config.items():
        if key not in ['output_dir', 'log_dir']:  # 这些在命令末尾
            cmd += f"    --{key} {value} \\\n"
    cmd += f"    --output_dir {repair_config['output_dir']} \\\n"
    cmd += f"    --log_dir {repair_config['log_dir']}"
    
    print(f"\n🚀 图像修复训练命令:")
    print(cmd)
    
    # 保存配置到文件
    with open('image_repair_config.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# 图像修复专用MAE训练配置\n")
        f.write("export KMP_DUPLICATE_LIB_OK=TRUE\n\n")
        f.write(cmd)
    
    os.chmod('image_repair_config.sh', 0o755)
    print(f"✅ 配置脚本保存: image_repair_config.sh")
    
    return repair_config

def main():
    """主函数"""
    print("🔍 MAE重建质量调试工具")
    print("=" * 50)
    
    # 1. 分析重建问题
    analyze_reconstruction_issues()
    
    # 2. 识别模糊原因
    identify_blur_causes()
    
    # 3. 测试不同归一化
    test_different_normalizations()
    
    # 4. 创建图像修复配置
    repair_config = create_image_repair_config()
    
    print(f"\n💡 解决模糊问题的建议:")
    print(f"  1️⃣ 🔴 增加训练轮数 (当前10 → 建议50-100)")
    print(f"  2️⃣ 🟡 降低学习率 (当前1.5e-4 → 建议5e-5)")
    print(f"  3️⃣ 🟡 延长预热期 (当前5 → 建议20 epochs)")
    print(f"  4️⃣ 🟢 使用更多训练数据")
    
    print(f"\n🎯 推荐立即尝试:")
    print(f"  bash image_repair_config.sh")

if __name__ == "__main__":
    main()


