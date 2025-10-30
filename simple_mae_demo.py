#!/usr/bin/env python3
"""
简单的MAE预训练模型演示脚本
展示如何加载和使用下载的预训练模型
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 检查是否安装了必要的包
try:
    import torch
    import torchvision.transforms as transforms
    print(f"✅ PyTorch 版本: {torch.__version__}")
except ImportError:
    print("❌ 请先安装 PyTorch:")
    print("pip install torch torchvision")
    sys.exit(1)

try:
    import models_mae
    print("✅ MAE 模型模块加载成功")
except ImportError:
    print("❌ 无法导入 models_mae 模块")
    sys.exit(1)

def load_pretrained_mae():
    """加载预训练的MAE模型"""
    print("\n🔄 加载预训练MAE模型...")
    
    # 检查设备
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✅ 使用 Apple Silicon MPS: {device}")
    else:
        device = torch.device('cpu')
        print(f"⚠️  使用 CPU: {device}")
    
    # 模型路径
    pretrain_path = 'pretrained_models/mae_pretrain_vit_base.pth'
    
    if not os.path.exists(pretrain_path):
        print(f"❌ 预训练模型不存在: {pretrain_path}")
        print("请先运行: ./download_models.sh")
        return None, None
    
    try:
        # 创建模型
        model = models_mae.mae_vit_base_patch16()
        
        # 加载预训练权重
        print(f"📥 从 {pretrain_path} 加载权重...")
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        
        # 移动到设备
        model = model.to(device)
        model.eval()
        
        print("✅ 预训练模型加载成功！")
        return model, device
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

def create_test_image():
    """创建一个测试图像"""
    print("\n🎨 创建测试图像...")
    
    # 创建一个简单的彩色测试图像
    img = np.zeros((224, 224, 3))
    
    # 添加一些几何图案
    # 红色矩形
    img[50:100, 50:150, 0] = 1.0
    
    # 绿色圆形
    y, x = np.ogrid[:224, :224]
    center_y, center_x = 112, 112
    mask = (x - center_x)**2 + (y - center_y)**2 <= 30**2
    img[mask, 1] = 1.0
    
    # 蓝色对角线
    for i in range(224):
        if i < 224:
            img[i, i, 2] = 1.0
            if i > 0:
                img[i-1, i, 2] = 0.5
            if i < 223:
                img[i+1, i, 2] = 0.5
    
    return img

def demonstrate_mae_reconstruction(model, device):
    """演示MAE重建过程"""
    print("\n🔍 演示MAE重建过程...")
    
    # 创建测试图像
    test_img = create_test_image()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 反归一化用于显示
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # 转换为tensor
    img_tensor = transform(test_img).unsqueeze(0).to(device)
    
    # 测试不同的掩码比例
    mask_ratios = [0.25, 0.5, 0.75]
    
    fig, axes = plt.subplots(len(mask_ratios), 4, figsize=(16, len(mask_ratios)*4))
    
    for i, mask_ratio in enumerate(mask_ratios):
        print(f"  测试掩码比例: {mask_ratio*100:.0f}%")
        
        with torch.no_grad():
            # MAE前向传播
            loss, pred, mask = model(img_tensor, mask_ratio=mask_ratio)
            
            # 重建图像
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
        axes[i, 0].set_title('原始图像')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(masked_img.permute(1, 2, 0))
        axes[i, 1].set_title(f'掩码图像 ({mask_ratio*100:.0f}%)')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(reconstructed_display.permute(1, 2, 0))
        axes[i, 2].set_title(f'重建图像\n损失: {loss.item():.3f}')
        axes[i, 2].axis('off')
        
        im = axes[i, 3].imshow(error_display, cmap='hot')
        axes[i, 3].set_title('重建误差')
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
        
        print(f"    损失值: {loss.item():.4f}")
        print(f"    实际掩码比例: {mask.float().mean().item():.2%}")
    
    plt.tight_layout()
    
    # 保存结果
    output_path = 'mae_pretrained_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 演示结果已保存: {output_path}")
    
    # 显示图像（如果在支持的环境中）
    try:
        plt.show()
    except:
        print("💡 如果要查看图像，请在支持图形界面的环境中运行")
    
    return output_path

def print_model_info(model):
    """打印模型信息"""
    print("\n📊 模型信息:")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  总参数量: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  可训练参数: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    
    # 模型结构信息
    print(f"  编码器层数: {model.depth}")
    print(f"  注意力头数: {model.num_heads}")
    print(f"  嵌入维度: {model.embed_dim}")
    print(f"  补丁大小: {model.patch_embed.patch_size}")

def main():
    """主函数"""
    print("🎭 MAE 预训练模型演示")
    print("=" * 50)
    
    # 设置环境变量
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 加载模型
    model, device = load_pretrained_mae()
    if model is None:
        return
    
    # 打印模型信息
    print_model_info(model)
    
    # 演示重建过程
    output_path = demonstrate_mae_reconstruction(model, device)
    
    print("\n🎉 演示完成！")
    print(f"📁 结果图像: {output_path}")
    print("\n💡 使用说明:")
    print("1. 这是使用Facebook官方预训练的ViT-Base MAE模型")
    print("2. 模型在ImageNet上预训练，具有强大的图像重建能力")
    print("3. 掩码比例越高，重建任务越困难")
    print("4. 可以用这个模型进行下游任务的微调")

if __name__ == "__main__":
    main()

