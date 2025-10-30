#!/usr/bin/env python3
"""
MAE编码器演示脚本
只使用预训练的编码器部分，展示特征提取能力
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

try:
    import torch
    import torchvision.transforms as transforms
    import models_mae
    print(f"✅ PyTorch 版本: {torch.__version__}")
except ImportError:
    print("❌ 请先安装 PyTorch: pip install torch torchvision")
    sys.exit(1)

def load_encoder_only():
    """加载预训练的MAE编码器"""
    print("\n🔄 加载预训练MAE编码器...")
    
    # 检查设备
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✅ 使用 Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        print(f"✅ 使用 CPU")
    
    # 模型路径
    pretrain_path = 'pretrained_models/mae_pretrain_vit_base.pth'
    
    if not os.path.exists(pretrain_path):
        print(f"❌ 预训练模型不存在: {pretrain_path}")
        return None, None
    
    try:
        # 创建完整模型
        model = models_mae.mae_vit_base_patch16()
        
        # 加载预训练权重
        print(f"📥 从 {pretrain_path} 加载权重...")
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        
        # 只加载编码器部分的权重
        encoder_state_dict = {}
        for key, value in checkpoint['model'].items():
            if not key.startswith('decoder') and key != 'mask_token':
                encoder_state_dict[key] = value
        
        # 使用strict=False来忽略解码器权重
        model.load_state_dict(encoder_state_dict, strict=False)
        
        # 移动到设备
        model = model.to(device)
        model.eval()
        
        print("✅ 预训练编码器加载成功！")
        return model, device
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

def create_test_images():
    """创建多个测试图像"""
    print("\n🎨 创建测试图像...")
    
    images = []
    
    # 图像1：几何图案
    img1 = np.zeros((224, 224, 3))
    # 红色矩形
    img1[50:100, 50:150, 0] = 1.0
    # 绿色圆形
    y, x = np.ogrid[:224, :224]
    center_y, center_x = 112, 112
    mask = (x - center_x)**2 + (y - center_y)**2 <= 30**2
    img1[mask, 1] = 1.0
    images.append(("几何图案", img1))
    
    # 图像2：渐变
    img2 = np.zeros((224, 224, 3))
    for i in range(224):
        img2[i, :, 0] = i / 224  # 红色渐变
        img2[:, i, 1] = i / 224  # 绿色渐变
    images.append(("彩色渐变", img2))
    
    # 图像3：棋盘格
    img3 = np.zeros((224, 224, 3))
    for i in range(0, 224, 32):
        for j in range(0, 224, 32):
            if (i//32 + j//32) % 2 == 0:
                img3[i:i+32, j:j+32] = 1.0
    images.append(("棋盘格", img3))
    
    return images

def extract_features(model, device, images):
    """提取图像特征"""
    print("\n🔍 提取图像特征...")
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    features_list = []
    
    for name, img in images:
        print(f"  处理图像: {name}")
        
        # 转换为tensor
        img_tensor = transform(img.astype(np.float32)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 只使用编码器部分
            # 图像分块
            x = model.patch_embed(img_tensor)
            
            # 添加位置编码
            x = x + model.pos_embed[:, 1:, :]
            
            # 添加cls token
            cls_token = model.cls_token + model.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            # 通过编码器
            for blk in model.blocks:
                x = blk(x)
            x = model.norm(x)
            
            # 提取特征
            cls_feature = x[:, 0]  # CLS token特征
            patch_features = x[:, 1:]  # patch特征
            
            features_list.append({
                'name': name,
                'cls_feature': cls_feature.cpu().numpy(),
                'patch_features': patch_features.cpu().numpy(),
                'original_image': img
            })
            
            print(f"    CLS特征维度: {cls_feature.shape}")
            print(f"    Patch特征维度: {patch_features.shape}")
    
    return features_list

def visualize_features(features_list):
    """可视化特征"""
    print("\n📊 可视化特征...")
    
    num_images = len(features_list)
    fig, axes = plt.subplots(3, num_images, figsize=(num_images*4, 12))
    
    if num_images == 1:
        axes = axes.reshape(-1, 1)
    
    for i, features in enumerate(features_list):
        # 原始图像
        axes[0, i].imshow(features['original_image'])
        axes[0, i].set_title(f"{features['name']}\n原始图像")
        axes[0, i].axis('off')
        
        # Patch特征的平均值可视化
        patch_features = features['patch_features'][0]  # (196, 768)
        patch_mean = patch_features.mean(axis=1)  # 每个patch的平均特征值
        patch_2d = patch_mean.reshape(14, 14)  # 重塑为14x14的特征图
        
        im1 = axes[1, i].imshow(patch_2d, cmap='viridis')
        axes[1, i].set_title('Patch特征均值')
        axes[1, i].axis('off')
        plt.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        # CLS特征的前64维可视化
        cls_feature = features['cls_feature'][0][:64]  # 取前64维
        cls_2d = cls_feature.reshape(8, 8)  # 重塑为8x8显示
        
        im2 = axes[2, i].imshow(cls_2d, cmap='coolwarm')
        axes[2, i].set_title('CLS特征 (前64维)')
        axes[2, i].axis('off')
        plt.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # 保存结果
    output_path = 'mae_encoder_features_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 特征可视化已保存: {output_path}")
    
    try:
        plt.show()
    except:
        print("💡 如果要查看图像，请在支持图形界面的环境中运行")
    
    return output_path

def analyze_features(features_list):
    """分析特征统计"""
    print("\n📈 特征分析:")
    
    for features in features_list:
        name = features['name']
        cls_feat = features['cls_feature'][0]
        patch_feat = features['patch_features'][0]
        
        print(f"\n  {name}:")
        print(f"    CLS特征统计:")
        print(f"      均值: {cls_feat.mean():.4f}")
        print(f"      标准差: {cls_feat.std():.4f}")
        print(f"      最大值: {cls_feat.max():.4f}")
        print(f"      最小值: {cls_feat.min():.4f}")
        
        print(f"    Patch特征统计:")
        print(f"      均值: {patch_feat.mean():.4f}")
        print(f"      标准差: {patch_feat.std():.4f}")
        print(f"      特征多样性: {patch_feat.std(axis=0).mean():.4f}")

def main():
    """主函数"""
    print("🎭 MAE 预训练编码器演示")
    print("=" * 50)
    
    # 设置环境变量
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 加载编码器
    model, device = load_encoder_only()
    if model is None:
        return
    
    # 创建测试图像
    test_images = create_test_images()
    
    # 提取特征
    features = extract_features(model, device, test_images)
    
    # 可视化特征
    output_path = visualize_features(features)
    
    # 分析特征
    analyze_features(features)
    
    print("\n🎉 演示完成！")
    print(f"📁 结果图像: {output_path}")
    print("\n💡 说明:")
    print("1. 这展示了预训练MAE编码器的特征提取能力")
    print("2. CLS token包含了整个图像的全局特征")
    print("3. Patch特征显示了每个16x16区域的局部特征")
    print("4. 这些特征可以用于下游任务如分类、检测等")

if __name__ == "__main__":
    main()
