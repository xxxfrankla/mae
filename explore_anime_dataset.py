#!/usr/bin/env python3
"""
探索动漫数据集
了解数据集结构和内容
"""

import os
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def explore_anime_dataset():
    """探索动漫数据集"""
    print("🎌 加载动漫数据集...")
    
    try:
        # 加载数据集
        ds = load_dataset("none-yet/anime-captions")
        
        print(f"✅ 数据集加载成功!")
        print(f"📊 数据集信息:")
        print(f"  - 数据集结构: {ds}")
        
        # 检查各个分割
        for split_name, split_data in ds.items():
            print(f"  - {split_name}: {len(split_data)} 样本")
            
            # 查看第一个样本
            if len(split_data) > 0:
                sample = split_data[0]
                print(f"    样本结构: {sample.keys()}")
                
                # 如果有图像，显示图像信息
                if 'image' in sample:
                    img = sample['image']
                    if isinstance(img, Image.Image):
                        print(f"    图像尺寸: {img.size}")
                        print(f"    图像模式: {img.mode}")
                
                # 如果有标题，显示标题
                if 'caption' in sample:
                    caption = sample['caption']
                    print(f"    标题示例: {caption[:100]}...")
                elif 'text' in sample:
                    text = sample['text']
                    print(f"    文本示例: {text[:100]}...")
        
        return ds
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return None

def visualize_anime_samples(ds, num_samples=6):
    """可视化动漫数据集样本"""
    print(f"\n🎨 可视化 {num_samples} 个动漫样本...")
    
    # 使用训练集或第一个可用的分割
    if 'train' in ds:
        split_data = ds['train']
    else:
        split_name = list(ds.keys())[0]
        split_data = ds[split_name]
        print(f"使用 {split_name} 分割")
    
    if len(split_data) == 0:
        print("❌ 数据集为空")
        return
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(split_data))):
        try:
            sample = split_data[i]
            
            if 'image' in sample:
                img = sample['image']
                if isinstance(img, Image.Image):
                    axes[i].imshow(img)
                    
                    # 添加标题
                    title = f"Sample {i+1}"
                    if 'caption' in sample:
                        caption = sample['caption']
                        if len(caption) > 50:
                            caption = caption[:47] + "..."
                        title += f"\n{caption}"
                    elif 'text' in sample:
                        text = sample['text']
                        if len(text) > 50:
                            text = text[:47] + "..."
                        title += f"\n{text}"
                    
                    axes[i].set_title(title, fontsize=10)
                    axes[i].axis('off')
                else:
                    axes[i].text(0.5, 0.5, f'Sample {i+1}\nNo Image', 
                               ha='center', va='center')
                    axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'Sample {i+1}\nNo Image Field', 
                           ha='center', va='center')
                axes[i].axis('off')
                
        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
            axes[i].text(0.5, 0.5, f'Sample {i+1}\nError: {str(e)[:30]}', 
                       ha='center', va='center')
            axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('anime_dataset_samples.png', dpi=150, bbox_inches='tight')
    print("✅ 样本可视化保存: anime_dataset_samples.png")
    plt.show()

def analyze_dataset_stats(ds):
    """分析数据集统计信息"""
    print(f"\n📈 数据集统计分析...")
    
    # 使用训练集或第一个可用的分割
    if 'train' in ds:
        split_data = ds['train']
        split_name = 'train'
    else:
        split_name = list(ds.keys())[0]
        split_data = ds[split_name]
    
    print(f"分析 {split_name} 分割...")
    
    # 分析图像尺寸
    image_sizes = []
    image_modes = []
    caption_lengths = []
    
    sample_size = min(1000, len(split_data))  # 分析前1000个样本
    print(f"分析前 {sample_size} 个样本...")
    
    for i in range(sample_size):
        try:
            sample = split_data[i]
            
            if 'image' in sample:
                img = sample['image']
                if isinstance(img, Image.Image):
                    image_sizes.append(img.size)
                    image_modes.append(img.mode)
            
            if 'caption' in sample:
                caption_lengths.append(len(sample['caption']))
            elif 'text' in sample:
                caption_lengths.append(len(sample['text']))
                
        except Exception as e:
            continue
    
    # 统计结果
    if image_sizes:
        widths = [size[0] for size in image_sizes]
        heights = [size[1] for size in image_sizes]
        
        print(f"\n🖼️  图像统计:")
        print(f"  样本数量: {len(image_sizes)}")
        print(f"  宽度范围: {min(widths)} - {max(widths)}")
        print(f"  高度范围: {min(heights)} - {max(heights)}")
        print(f"  平均尺寸: {np.mean(widths):.1f} x {np.mean(heights):.1f}")
        
        # 统计常见尺寸
        size_counts = {}
        for size in image_sizes:
            size_counts[size] = size_counts.get(size, 0) + 1
        
        print(f"  最常见的5个尺寸:")
        for size, count in sorted(size_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {size[0]}x{size[1]}: {count} 张 ({count/len(image_sizes)*100:.1f}%)")
    
    if image_modes:
        mode_counts = {}
        for mode in image_modes:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        print(f"\n🎨 图像模式:")
        for mode, count in mode_counts.items():
            print(f"  {mode}: {count} 张 ({count/len(image_modes)*100:.1f}%)")
    
    if caption_lengths:
        print(f"\n📝 标题统计:")
        print(f"  平均长度: {np.mean(caption_lengths):.1f} 字符")
        print(f"  长度范围: {min(caption_lengths)} - {max(caption_lengths)}")

def main():
    """主函数"""
    print("🎌 动漫数据集探索工具")
    print("=" * 50)
    
    # 1. 探索数据集
    ds = explore_anime_dataset()
    
    if ds is None:
        print("❌ 无法加载数据集，退出")
        return
    
    # 2. 可视化样本
    try:
        visualize_anime_samples(ds)
    except Exception as e:
        print(f"可视化失败: {e}")
    
    # 3. 分析统计信息
    try:
        analyze_dataset_stats(ds)
    except Exception as e:
        print(f"统计分析失败: {e}")
    
    print(f"\n🎉 数据集探索完成!")
    print(f"💡 下一步可以:")
    print(f"  1. 创建适配的数据加载器")
    print(f"  2. 在动漫数据集上运行MAE预训练")
    print(f"  3. 对比不同数据集的训练效果")

if __name__ == "__main__":
    main()
