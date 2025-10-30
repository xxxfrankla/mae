#!/usr/bin/env python3
"""
详细解释MAE中Reconstruction和Reconstruction + Visible的区别
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import models_mae
import requests

# 设置环境
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def load_complete_mae_model():
    """加载完整的MAE模型"""
    print("🤖 加载完整MAE模型...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = models_mae.mae_vit_large_patch16()
    
    model_path = 'complete_mae_models/mae_visualize_vit_large.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        model = model.to(device)
        model.eval()
        print("✅ 模型加载成功")
        return model, device
    else:
        print("❌ 模型文件不存在")
        return None, None

def explain_reconstruction_types():
    """详细解释重建类型的区别"""
    print("\n📚 MAE重建类型详解:")
    print("=" * 60)
    
    print("🎯 1. Reconstruction (纯重建)")
    print("   • 定义: 模型对所有patches的预测结果")
    print("   • 内容: 包括被mask的部分 + 可见部分的预测")
    print("   • 特点: 完全由模型生成，可能与原图有差异")
    print("   • 公式: y = model.unpatchify(pred)")
    print()
    
    print("🎯 2. Reconstruction + Visible (重建+可见)")
    print("   • 定义: 原始可见部分 + 重建的mask部分")
    print("   • 内容: 保留原图可见patches，只显示重建的mask部分")
    print("   • 特点: 可见部分是完美的原图，mask部分是重建")
    print("   • 公式: result = original * (1-mask) + reconstruction * mask")
    print()
    
    print("💡 关键区别:")
    print("   Reconstruction: 模型对整张图的'理解'和'重建'")
    print("   Reconstruction + Visible: 实际应用中的'修复'效果")

def demonstrate_reconstruction_difference(model, device):
    """演示两种重建方式的区别"""
    print("\n🎨 演示重建差异...")
    
    # ImageNet标准化参数
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    # 加载测试图像
    img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg'
    img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.resize((224, 224))
    img = np.array(img) / 255.
    
    # 标准化
    img_normalized = (img - imagenet_mean) / imagenet_std
    
    # 转换为tensor
    x = torch.tensor(img_normalized).float()
    x = x.unsqueeze(0).permute(0, 3, 1, 2).to(device)  # (1, 3, 224, 224)
    
    # 设置随机种子确保可重现
    torch.manual_seed(2)
    
    # MAE前向传播
    with torch.no_grad():
        loss, pred, mask = model(x, mask_ratio=0.75)
        
        # 1. 纯重建 (Reconstruction)
        reconstruction_full = model.unpatchify(pred)  # 模型对所有patches的预测
        
        # 2. 创建mask可视化
        mask_vis = mask.detach()
        mask_vis = mask_vis.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
        mask_vis = model.unpatchify(mask_vis)  # 1表示被mask，0表示可见
        
        # 3. 重建+可见 (Reconstruction + Visible)
        reconstruction_plus_visible = x * (1 - mask_vis) + reconstruction_full * mask_vis
        
        # 4. 掩码图像 (用于对比)
        masked_image = x * (1 - mask_vis) + mask_vis * 0.5  # 灰色表示被mask区域
    
    # 反标准化用于显示
    def denormalize(tensor):
        tensor = tensor.cpu().permute(0, 2, 3, 1)[0]  # (H, W, 3)
        tensor = tensor * torch.tensor(imagenet_std) + torch.tensor(imagenet_mean)
        return torch.clamp(tensor, 0, 1)
    
    # 创建对比可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 第一行：基础对比
    axes[0, 0].imshow(denormalize(x))
    axes[0, 0].set_title('Original Image\n(原始图像)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(denormalize(masked_image))
    axes[0, 1].set_title('Masked Image (75%)\n(掩码图像)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(denormalize(reconstruction_plus_visible))
    axes[0, 2].set_title('Reconstruction + Visible\n(重建+可见)\n✅ 实际应用效果', fontsize=14, fontweight='bold', color='green')
    axes[0, 2].axis('off')
    
    # 第二行：详细分析
    axes[1, 0].imshow(denormalize(reconstruction_full))
    axes[1, 0].set_title('Pure Reconstruction\n(纯重建)\n🤖 模型的完整预测', fontsize=14, fontweight='bold', color='blue')
    axes[1, 0].axis('off')
    
    # 显示差异
    diff_visible = torch.abs(denormalize(x) - denormalize(reconstruction_full))
    axes[1, 1].imshow(diff_visible, cmap='hot')
    axes[1, 1].set_title('Difference: Original vs Reconstruction\n(原图 vs 纯重建的差异)\n🔍 可见部分的预测误差', fontsize=14, fontweight='bold', color='red')
    axes[1, 1].axis('off')
    
    # 只显示重建的mask部分
    mask_only_reconstruction = denormalize(reconstruction_full * mask_vis + (1 - mask_vis) * 0.5)
    axes[1, 2].imshow(mask_only_reconstruction)
    axes[1, 2].set_title('Mask Area Reconstruction Only\n(仅重建区域)\n🎨 模型重建的mask部分', fontsize=14, fontweight='bold', color='purple')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    output_path = 'reconstruction_difference_explanation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 对比图保存: {output_path}")
    
    plt.show()
    
    # 打印统计信息
    print(f"\n📊 统计信息:")
    print(f"  损失值: {loss.item():.4f}")
    print(f"  掩码比例: {mask.float().mean().item():.2%}")
    
    # 计算可见部分的重建误差
    visible_area = (1 - mask_vis).bool()
    if visible_area.sum() > 0:
        visible_error = torch.abs(x - reconstruction_full)[visible_area].mean()
        print(f"  可见部分重建误差: {visible_error.item():.4f}")
    
    return output_path

def create_detailed_explanation():
    """创建详细的文字解释"""
    print("\n📖 详细技术解释:")
    print("=" * 60)
    
    print("🔍 技术细节:")
    print()
    print("1️⃣ Reconstruction (纯重建):")
    print("   • 来源: model.unpatchify(pred)")
    print("   • 含义: 模型基于25%可见patches对整张图的预测")
    print("   • 特点: 可见部分也是预测的，可能与原图不完全一致")
    print("   • 用途: 评估模型的理解和生成能力")
    print()
    
    print("2️⃣ Reconstruction + Visible (重建+可见):")
    print("   • 来源: original * (1-mask) + reconstruction * mask")
    print("   • 含义: 保留原图可见部分，只替换重建的mask部分")
    print("   • 特点: 可见部分完美，只有mask部分是重建的")
    print("   • 用途: 图像修复、去噪、内容填充等实际应用")
    print()
    
    print("🎯 应用场景:")
    print("   • Reconstruction: 研究模型能力，学术分析")
    print("   • Reconstruction + Visible: 实际产品应用，用户体验")
    print()
    
    print("💡 为什么需要两种显示方式？")
    print("   1. 纯重建显示模型的'想象力'和'理解力'")
    print("   2. 重建+可见显示实际应用效果")
    print("   3. 对比两者可以分析模型在可见部分的预测准确性")

def main():
    """主函数"""
    print("🎭 MAE重建类型差异详解")
    print("=" * 50)
    
    # 理论解释
    explain_reconstruction_types()
    
    # 加载模型
    model, device = load_complete_mae_model()
    if model is None:
        return
    
    # 实际演示
    output_path = demonstrate_reconstruction_difference(model, device)
    
    # 详细解释
    create_detailed_explanation()
    
    print(f"\n🎉 解释完成!")
    print(f"📁 对比图: {output_path}")
    print(f"\n🎯 总结:")
    print(f"  Reconstruction: 模型的完整预测 (学术研究用)")
    print(f"  Reconstruction + Visible: 实际修复效果 (产品应用用)")

if __name__ == "__main__":
    main()
