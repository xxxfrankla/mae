#!/usr/bin/env python3
"""
动漫数据集加载器
适配 HuggingFace datasets 到 MAE 训练流程
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class AnimeDataset(Dataset):
    """动漫数据集包装器"""
    
    def __init__(self, hf_dataset, transform=None, max_samples=None):
        """
        Args:
            hf_dataset: HuggingFace 数据集
            transform: 图像变换
            max_samples: 最大样本数（用于测试）
        """
        self.dataset = hf_dataset
        self.transform = transform
        
        # 限制样本数量（用于快速测试）
        if max_samples is not None:
            self.length = min(max_samples, len(hf_dataset))
        else:
            self.length = len(hf_dataset)
        
        print(f"📊 动漫数据集: {self.length} 张图片")

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
                # 如果是numpy数组，转换为PIL图像
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    # 创建一个默认图像
                    image = Image.new('RGB', (512, 512), color='black')
            
            # 确保是RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            # MAE 不需要标签，返回 0 作为占位符
            return image, 0
            
        except Exception as e:
            print(f"警告: 加载样本 {idx} 时出错: {e}")
            # 返回一个默认图像
            default_img = Image.new('RGB', (512, 512), color='black')
            if self.transform:
                default_img = self.transform(default_img)
            return default_img, 0

def create_anime_dataloader(batch_size=8, num_workers=4, max_samples=None, input_size=224):
    """创建动漫数据集的数据加载器"""
    
    print(f"🎌 创建动漫数据集加载器...")
    print(f"  批次大小: {batch_size}")
    print(f"  输入尺寸: {input_size}x{input_size}")
    if max_samples:
        print(f"  最大样本数: {max_samples}")
    
    # 加载 HuggingFace 数据集
    try:
        ds = load_dataset("none-yet/anime-captions")
        train_dataset = ds['train']
        print(f"✅ HuggingFace 数据集加载成功")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return None
    
    # 定义图像变换（与 MAE 原始设置一致）
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建包装数据集
    anime_dataset = AnimeDataset(train_dataset, transform=transform, max_samples=max_samples)
    
    # 创建数据加载器
    dataloader = DataLoader(
        anime_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # MPS 不支持 pin_memory
        drop_last=True
    )
    
    print(f"✅ 数据加载器创建成功")
    print(f"  总批次数: {len(dataloader)}")
    
    return dataloader, anime_dataset

def test_anime_dataloader():
    """测试动漫数据加载器"""
    print("🧪 测试动漫数据加载器...")
    
    # 创建小规模测试加载器
    dataloader, dataset = create_anime_dataloader(
        batch_size=4, 
        max_samples=100,  # 只测试100张图片
        input_size=224
    )
    
    if dataloader is None:
        return
    
    # 测试加载几个批次
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    for i, (images, labels) in enumerate(dataloader):
        print(f"  批次 {i+1}:")
        print(f"    图像形状: {images.shape}")
        print(f"    图像范围: [{images.min():.3f}, {images.max():.3f}]")
        print(f"    标签: {labels}")
        
        # 测试在设备上运行
        images = images.to(device)
        print(f"    设备: {images.device}")
        
        if i >= 2:  # 只测试前3个批次
            break
    
    print("✅ 数据加载器测试通过!")

def main():
    """主函数"""
    print("🎌 动漫数据集加载器测试")
    print("=" * 50)
    
    # 设置环境
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 测试数据加载器
    test_anime_dataloader()

if __name__ == "__main__":
    main()


