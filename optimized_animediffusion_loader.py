#!/usr/bin/env python3
"""
优化的 AnimeDiffusion 数据集加载器
将高分辨率图像预处理到合适尺寸
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class OptimizedAnimeDiffusionDataset(Dataset):
    """优化的 AnimeDiffusion 数据集包装器"""
    
    def __init__(self, hf_dataset, transform=None, max_samples=None, target_size=224):
        """
        Args:
            hf_dataset: HuggingFace 数据集
            transform: 图像变换
            max_samples: 最大样本数
            target_size: 目标图像尺寸
        """
        self.dataset = hf_dataset
        self.transform = transform
        self.target_size = target_size
        
        if max_samples is not None:
            self.length = min(max_samples, len(hf_dataset))
        else:
            self.length = len(hf_dataset)
        
        print(f"📊 优化的 AnimeDiffusion 数据集:")
        print(f"  原始分辨率: 1920×1080")
        print(f"  目标分辨率: {target_size}×{target_size}")
        print(f"  样本数量: {self.length}")
        print(f"  分辨率缩放: {1920/target_size:.1f}x 缩小")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """获取单个样本"""
        try:
            sample = self.dataset[idx]
            
            # 获取图像
            image = sample['image']
            
            # 确保是PIL图像
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    image = Image.new('RGB', (1920, 1080), color='black')
            
            # 处理RGBA图像
            if image.mode == 'RGBA':
                # 创建白色背景
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            return image, 0
            
        except Exception as e:
            print(f"警告: 加载样本 {idx} 时出错: {e}")
            # 返回默认图像
            default_img = Image.new('RGB', (self.target_size, self.target_size), color='black')
            if self.transform:
                default_img = self.transform(default_img)
            return default_img, 0

def create_optimized_animediffusion_dataloader(batch_size=8, num_workers=4, max_samples=None, 
                                             input_size=224, resize_strategy='smart_crop'):
    """
    创建优化的 AnimeDiffusion 数据加载器
    
    Args:
        resize_strategy: 'smart_crop', 'center_crop', 'resize_only'
    """
    
    print(f"🎌 创建优化的 AnimeDiffusion 数据集加载器...")
    print(f"  批次大小: {batch_size}")
    print(f"  输入尺寸: {input_size}x{input_size}")
    print(f"  缩放策略: {resize_strategy}")
    if max_samples:
        print(f"  最大样本数: {max_samples}")
    
    # 加载数据集
    try:
        ds = load_dataset("Mercity/AnimeDiffusion_Dataset")
        train_dataset = ds['train']
        print(f"✅ AnimeDiffusion 数据集加载成功 ({len(train_dataset)} 张图片)")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return None, None
    
    # 根据策略定义不同的变换
    if resize_strategy == 'smart_crop':
        # 智能裁剪：保持宽高比，然后随机裁剪
        transform = transforms.Compose([
            transforms.Resize(int(input_size * 1.2), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif resize_strategy == 'center_crop':
        # 中心裁剪：先resize再中心裁剪
        transform = transforms.Compose([
            transforms.Resize(int(input_size * 1.1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # resize_only
        # 直接缩放：可能会变形但保持完整内容
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # 创建数据集
    anime_dataset = OptimizedAnimeDiffusionDataset(
        train_dataset, 
        transform=transform, 
        max_samples=max_samples,
        target_size=input_size
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        anime_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True
    )
    
    print(f"✅ 优化的数据加载器创建成功")
    print(f"  总批次数: {len(dataloader)}")
    
    return dataloader, anime_dataset

def compare_resize_strategies():
    """对比不同的缩放策略"""
    print("🔍 对比不同的图像缩放策略...")
    
    strategies = ['smart_crop', 'center_crop', 'resize_only']
    
    fig, axes = plt.subplots(len(strategies), 3, figsize=(12, len(strategies)*4))
    
    # 加载一张测试图片
    try:
        ds = load_dataset("Mercity/AnimeDiffusion_Dataset")
        sample = ds['train'][0]
        original_img = sample['image']
        
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        
        print(f"原始图像尺寸: {original_img.size}")
        
        for i, strategy in enumerate(strategies):
            # 创建对应的变换
            if strategy == 'smart_crop':
                transform = transforms.Compose([
                    transforms.Resize(int(224 * 1.2), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),  # 用中心裁剪代替随机裁剪用于演示
                    transforms.ToTensor()
                ])
            elif strategy == 'center_crop':
                transform = transforms.Compose([
                    transforms.Resize(int(224 * 1.1), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                ])
            else:  # resize_only
                transform = transforms.Compose([
                    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor()
                ])
            
            # 应用变换
            processed_img = transform(original_img)
            
            # 显示原图（只在第一行显示）
            if i == 0:
                # 缩小原图用于显示
                display_original = original_img.resize((224, 126))  # 保持16:9比例
                axes[i, 0].imshow(display_original)
                axes[i, 0].set_title('Original\n1920×1080')
                axes[i, 0].axis('off')
            else:
                axes[i, 0].axis('off')
            
            # 显示处理后的图像
            axes[i, 1].imshow(processed_img.permute(1, 2, 0))
            axes[i, 1].set_title(f'{strategy}\n224×224')
            axes[i, 1].axis('off')
            
            # 显示策略说明
            if strategy == 'smart_crop':
                description = "先缩放到1.2倍\n然后随机/中心裁剪\n保持细节，可能丢失边缘"
            elif strategy == 'center_crop':
                description = "先缩放到1.1倍\n然后中心裁剪\n保持中心内容"
            else:
                description = "直接缩放到目标尺寸\n可能变形但保持完整内容"
            
            axes[i, 2].text(0.1, 0.5, description, transform=axes[i, 2].transAxes,
                          fontsize=10, verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[i, 2].axis('off')
            axes[i, 2].set_title(f'{strategy} 说明')
        
        plt.tight_layout()
        plt.savefig('resize_strategy_comparison.png', dpi=150, bbox_inches='tight')
        print("✅ 缩放策略对比保存: resize_strategy_comparison.png")
        plt.close()
        
    except Exception as e:
        print(f"对比失败: {e}")

def test_memory_usage():
    """测试不同分辨率的内存使用"""
    print("\n💾 测试不同分辨率的内存使用...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 测试不同的输入尺寸
    input_sizes = [224, 256, 288, 320]
    batch_sizes = [8, 6, 4, 2]
    
    import models_mae
    
    for input_size, batch_size in zip(input_sizes, batch_sizes):
        try:
            print(f"\n🧪 测试 {input_size}×{input_size}, batch_size={batch_size}")
            
            # 创建模型
            model = models_mae.mae_vit_base_patch16()
            model.to(device)
            model.eval()
            
            # 创建测试数据
            x = torch.randn(batch_size, 3, input_size, input_size, device=device)
            
            # 测试前向传播
            with torch.no_grad():
                loss, pred, mask = model(x, mask_ratio=0.75)
            
            print(f"  ✅ 成功: 损失 {loss.item():.4f}")
            
            # 清理内存
            del model, x, loss, pred, mask
            if device.type == 'mps':
                torch.mps.empty_cache()
                
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            # 清理内存
            if device.type == 'mps':
                torch.mps.empty_cache()

def main():
    """主函数"""
    print("🎌 AnimeDiffusion 分辨率优化分析")
    print("=" * 50)
    
    # 设置环境
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 1. 测试数据加载器
    test_animediffusion_dataloader()
    
    # 2. 对比缩放策略
    try:
        import matplotlib.pyplot as plt
        compare_resize_strategies()
    except Exception as e:
        print(f"缩放策略对比失败: {e}")
    
    # 3. 测试内存使用
    test_memory_usage()
    
    print(f"\n💡 建议:")
    print(f"  - 推荐输入尺寸: 224×224 (标准)")
    print(f"  - 可尝试: 256×256 (更多细节)")
    print(f"  - 避免: >320×320 (内存不足)")
    print(f"  - 推荐策略: smart_crop (保持细节)")

if __name__ == "__main__":
    main()


