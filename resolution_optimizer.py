#!/usr/bin/env python3
"""
AnimeDiffusion 分辨率优化器
测试不同分辨率和策略的效果
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

# 解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import models_mae

def test_different_resolutions():
    """测试不同分辨率的内存和性能"""
    print("💾 测试不同分辨率的内存和性能...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 测试配置
    test_configs = [
        {'size': 224, 'batch': 8, 'desc': '标准配置'},
        {'size': 256, 'batch': 6, 'desc': '中等分辨率'},
        {'size': 288, 'batch': 4, 'desc': '高分辨率'},
        {'size': 320, 'batch': 2, 'desc': '极高分辨率'},
    ]
    
    results = []
    
    for config in test_configs:
        size = config['size']
        batch_size = config['batch']
        desc = config['desc']
        
        try:
            print(f"\n🧪 测试 {desc}: {size}×{size}, batch_size={batch_size}")
            
            # 创建模型
            model = models_mae.mae_vit_base_patch16()
            model.to(device)
            model.eval()
            
            # 创建测试数据
            x = torch.randn(batch_size, 3, size, size, device=device)
            
            # 预热
            with torch.no_grad():
                _ = model(x, mask_ratio=0.75)
            
            # 计时测试
            start_time = time.time()
            num_runs = 5
            
            for _ in range(num_runs):
                with torch.no_grad():
                    loss, pred, mask = model(x, mask_ratio=0.75)
            
            if device.type == 'mps':
                torch.mps.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs
            time_per_image = avg_time / batch_size
            
            results.append({
                'size': size,
                'batch_size': batch_size,
                'desc': desc,
                'avg_time': avg_time,
                'time_per_image': time_per_image,
                'loss': loss.item(),
                'success': True
            })
            
            print(f"  ✅ 成功: {avg_time*1000:.1f}ms/batch, {time_per_image*1000:.1f}ms/image")
            print(f"  损失: {loss.item():.4f}")
            
            # 清理内存
            del model, x, loss, pred, mask
            if device.type == 'mps':
                torch.mps.empty_cache()
                
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            results.append({
                'size': size,
                'batch_size': batch_size,
                'desc': desc,
                'success': False,
                'error': str(e)
            })
            
            # 清理内存
            if device.type == 'mps':
                torch.mps.empty_cache()
    
    # 显示结果总结
    print(f"\n📊 性能测试总结:")
    print(f"{'分辨率':<10} {'批次大小':<8} {'状态':<6} {'时间/图片':<12} {'描述'}")
    print("-" * 60)
    
    for result in results:
        if result['success']:
            status = "✅"
            time_str = f"{result['time_per_image']*1000:.1f}ms"
        else:
            status = "❌"
            time_str = "失败"
        
        print(f"{result['size']}×{result['size']:<3} {result['batch_size']:<8} {status:<6} {time_str:<12} {result['desc']}")
    
    return results

def create_resolution_comparison():
    """创建分辨率对比可视化"""
    print(f"\n🎨 创建分辨率对比可视化...")
    
    try:
        # 加载一张示例图片
        ds = load_dataset("Mercity/AnimeDiffusion_Dataset")
        sample = ds['train'][0]
        original_img = sample['image']
        
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        
        # 测试不同分辨率
        resolutions = [224, 256, 288, 320]
        
        fig, axes = plt.subplots(2, len(resolutions), figsize=(len(resolutions)*4, 8))
        
        for i, res in enumerate(resolutions):
            # 智能裁剪策略
            transform_crop = transforms.Compose([
                transforms.Resize(int(res * 1.2), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(res),
                transforms.ToTensor()
            ])
            
            # 直接缩放策略
            transform_resize = transforms.Compose([
                transforms.Resize((res, res), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor()
            ])
            
            # 应用变换
            img_crop = transform_crop(original_img)
            img_resize = transform_resize(original_img)
            
            # 显示结果
            axes[0, i].imshow(img_crop.permute(1, 2, 0))
            axes[0, i].set_title(f'Smart Crop\n{res}×{res}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(img_resize.permute(1, 2, 0))
            axes[1, i].set_title(f'Direct Resize\n{res}×{res}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('resolution_comparison.png', dpi=150, bbox_inches='tight')
        print("✅ 分辨率对比保存: resolution_comparison.png")
        plt.close()
        
    except Exception as e:
        print(f"分辨率对比失败: {e}")

def recommend_optimal_config():
    """推荐最优配置"""
    print(f"\n🎯 推荐最优配置:")
    
    # 基于Apple M4 24GB的配置建议
    configs = {
        'quick_test': {
            'input_size': 224,
            'batch_size': 8,
            'max_samples': 500,
            'epochs': 5,
            'description': '快速测试 - 5分钟验证'
        },
        'balanced': {
            'input_size': 224,
            'batch_size': 6,
            'max_samples': 2000,
            'epochs': 20,
            'description': '平衡配置 - 30分钟训练'
        },
        'high_quality': {
            'input_size': 256,
            'batch_size': 4,
            'max_samples': 5000,
            'epochs': 50,
            'description': '高质量 - 2小时训练'
        },
        'full_dataset': {
            'input_size': 224,
            'batch_size': 8,
            'max_samples': None,  # 全部8202张
            'epochs': 100,
            'description': '完整数据集 - 数小时训练'
        }
    }
    
    print(f"{'配置':<15} {'分辨率':<8} {'批次':<6} {'样本数':<8} {'轮数':<6} {'描述'}")
    print("-" * 70)
    
    for name, config in configs.items():
        samples_str = str(config['max_samples']) if config['max_samples'] else 'All'
        print(f"{name:<15} {config['input_size']}×{config['input_size']:<3} {config['batch_size']:<6} {samples_str:<8} {config['epochs']:<6} {config['description']}")
    
    return configs

def main():
    """主函数"""
    print("🎌 AnimeDiffusion 分辨率优化分析")
    print("=" * 50)
    
    # 1. 测试不同分辨率的性能
    performance_results = test_different_resolutions()
    
    # 2. 创建分辨率对比
    create_resolution_comparison()
    
    # 3. 推荐最优配置
    configs = recommend_optimal_config()
    
    # 4. 生成训练命令
    print(f"\n🚀 推荐的训练命令:")
    
    print(f"\n1️⃣ 快速测试 (推荐先试):")
    print(f"python main_pretrain_animediffusion.py --input_size 224 --batch_size 8 --max_samples 500 --epochs 5")
    
    print(f"\n2️⃣ 平衡训练:")
    print(f"python main_pretrain_animediffusion.py --input_size 224 --batch_size 6 --max_samples 2000 --epochs 20")
    
    print(f"\n3️⃣ 高质量训练:")
    print(f"python main_pretrain_animediffusion.py --input_size 256 --batch_size 4 --max_samples 5000 --epochs 50")
    
    print(f"\n💡 关键建议:")
    print(f"  ✅ 使用 224×224 分辨率 (内存友好)")
    print(f"  ✅ 使用 smart_crop 策略 (保持细节)")
    print(f"  ✅ 从小样本开始测试")
    print(f"  ⚠️  避免直接使用 1920×1080 (内存溢出)")

if __name__ == "__main__":
    main()


