#!/usr/bin/env python3
"""
MAE 在 Apple M4 上的测试脚本
测试 MPS 支持、模型加载和基本推理功能
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import models_mae
import models_vit

def test_environment():
    """测试环境配置"""
    print("🔍 环境检测")
    print(f"✅ PyTorch 版本: {torch.__version__}")
    print(f"✅ MPS 可用: {torch.backends.mps.is_available()}")
    print(f"✅ MPS 构建: {torch.backends.mps.is_built()}")
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        x = torch.randn(100, 100, device=device)
        y = x @ x.t()
        print(f"✅ MPS 测试通过: {y.device}")
        return device
    else:
        print("⚠️  MPS 不可用，使用 CPU")
        return torch.device('cpu')

def test_mae_models(device):
    """测试 MAE 模型"""
    print("\n🤖 MAE 模型测试")
    
    # 测试不同规模的模型
    models_to_test = [
        ('ViT-Base', models_mae.mae_vit_base_patch16),
        ('ViT-Large', models_mae.mae_vit_large_patch16),
    ]
    
    for name, model_func in models_to_test:
        try:
            print(f"\n📊 测试 {name}")
            model = model_func()
            model = model.to(device)
            
            # 计算参数量
            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  参数量: {params:.1f}M")
            
            # 测试前向传播
            x = torch.randn(1, 3, 224, 224, device=device)
            with torch.no_grad():
                loss, pred, mask = model(x, mask_ratio=0.75)
            
            print(f"  ✅ 前向传播成功")
            print(f"  损失值: {loss.item():.4f}")
            print(f"  预测形状: {pred.shape}")
            print(f"  掩码形状: {mask.shape}")
            
        except Exception as e:
            print(f"  ❌ {name} 测试失败: {e}")

def test_vit_models(device):
    """测试 ViT 分类模型"""
    print("\n🎯 ViT 分类模型测试")
    
    try:
        model = models_vit.vit_base_patch16(num_classes=1000)
        model = model.to(device)
        
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  参数量: {params:.1f}M")
        
        # 测试分类
        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            logits = model(x)
        
        print(f"  ✅ 分类测试成功")
        print(f"  输出形状: {logits.shape}")
        print(f"  预测类别: {logits.argmax(dim=1).cpu().numpy()}")
        
    except Exception as e:
        print(f"  ❌ ViT 测试失败: {e}")

def test_memory_usage(device):
    """测试内存使用情况"""
    print("\n💾 内存使用测试")
    
    if device.type == 'mps':
        # 测试不同 batch size 的内存使用
        batch_sizes = [1, 2, 4, 8]
        model = models_mae.mae_vit_base_patch16().to(device)
        
        for bs in batch_sizes:
            try:
                x = torch.randn(bs, 3, 224, 224, device=device)
                with torch.no_grad():
                    loss, pred, mask = model(x, mask_ratio=0.75)
                print(f"  ✅ Batch size {bs}: 成功")
                del x, loss, pred, mask
                torch.mps.empty_cache()
            except Exception as e:
                print(f"  ❌ Batch size {bs}: {e}")
                break
    else:
        print("  跳过 MPS 内存测试（设备不支持）")

def create_demo_visualization():
    """创建演示可视化"""
    print("\n🎨 创建演示可视化")
    
    try:
        # 创建一个简单的演示图像
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像（随机彩色图像）
        original = np.random.rand(224, 224, 3)
        axes[0].imshow(original)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 掩码图像（75% 被掩盖）
        masked = original.copy()
        mask = np.random.rand(224, 224) < 0.75
        masked[mask] = 0.5  # 灰色表示被掩盖的区域
        axes[1].imshow(masked)
        axes[1].set_title('掩码图像 (75% 掩盖)')
        axes[1].axis('off')
        
        # 重建图像（模拟）
        reconstructed = original + np.random.normal(0, 0.1, original.shape)
        reconstructed = np.clip(reconstructed, 0, 1)
        axes[2].imshow(reconstructed)
        axes[2].set_title('重建图像')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('/Users/tdu/Documents/GitHub/mae/mae_demo_m4.png', dpi=150, bbox_inches='tight')
        print("  ✅ 演示图像已保存: mae_demo_m4.png")
        plt.close()
        
    except Exception as e:
        print(f"  ❌ 可视化创建失败: {e}")

def main():
    """主测试函数"""
    print("🍎 MAE Apple M4 24GB 兼容性测试")
    print("=" * 50)
    
    # 环境测试
    device = test_environment()
    
    # 模型测试
    test_mae_models(device)
    test_vit_models(device)
    
    # 内存测试
    test_memory_usage(device)
    
    # 创建演示
    create_demo_visualization()
    
    print("\n🎉 测试完成！")
    print("\n📝 使用建议:")
    print("1. 使用 batch_size=8 或更小以适应 24GB 内存")
    print("2. 运行前执行: source setup_env.sh")
    print("3. 可以运行 jupyter notebook demo/mae_visualize.ipynb 查看交互式演示")
    print("4. 下载预训练模型进行微调和评估")

if __name__ == "__main__":
    main()
