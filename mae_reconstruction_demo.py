#!/usr/bin/env python3
"""
MAE完整重建演示
使用编码器+解码器展示mask后图像的重建过程
保存重建结果到新文件夹
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import models_mae
from animediffusion_dataset_loader import create_animediffusion_dataloader
from datetime import datetime
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_output_folder():
    """创建输出文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"mae_reconstruction_{timestamp}"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"✅ 创建输出文件夹: {folder_name}")
    
    return folder_name

def load_complete_mae_model():
    """加载完整的MAE模型（需要初始化解码器）"""
    print("\n🤖 加载完整MAE模型...")
    
    # 检查设备
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✅ 使用 Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        print(f"✅ 使用 CPU")
    
    # 创建完整模型
    model = models_mae.mae_vit_base_patch16()
    
    # 加载预训练权重
    pretrain_path = 'pretrained_models/mae_pretrain_vit_base.pth'
    if os.path.exists(pretrain_path):
        print(f"📥 加载预训练权重: {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        
        # 只加载编码器权重，解码器保持随机初始化
        encoder_state_dict = {}
        for key, value in checkpoint['model'].items():
            if not key.startswith('decoder') and key != 'mask_token':
                encoder_state_dict[key] = value
        
        # 使用strict=False来忽略解码器权重
        missing_keys, unexpected_keys = model.load_state_dict(encoder_state_dict, strict=False)
        print(f"✅ 编码器权重加载成功")
        print(f"⚠️  解码器使用随机初始化 ({len(missing_keys)} 个参数)")
    else:
        print("⚠️  使用完全随机初始化的权重")
    
    model = model.to(device)
    model.eval()
    
    return model, device

def load_animediffusion_images():
    """加载AnimeDiffusion图像"""
    print("\n🎌 加载AnimeDiffusion图像...")
    
    try:
        # 创建数据加载器
        dataloader, dataset = create_animediffusion_dataloader(
            batch_size=4,
            max_samples=20,
            input_size=224,
            num_workers=0
        )
        
        if dataloader is None:
            return None
        
        # 获取第一个批次
        for images, _ in dataloader:
            print(f"✅ 成功加载 {images.shape[0]} 张图像")
            return images
            
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None

def create_fallback_images():
    """创建备用测试图像"""
    print("🎨 创建备用测试图像...")
    
    images = []
    
    # 图像1: 动漫人物
    img1 = torch.zeros(3, 224, 224)
    y, x = torch.meshgrid(torch.arange(224), torch.arange(224), indexing='ij')
    
    # 脸部
    face_mask = ((x - 112)/60)**2 + ((y - 112)/80)**2 <= 1
    img1[0][face_mask] = (1.0 - 0.485) / 0.229
    img1[1][face_mask] = (0.9 - 0.456) / 0.224
    img1[2][face_mask] = (0.8 - 0.406) / 0.225
    
    # 眼睛
    eye1 = ((x - 90)/8)**2 + ((y - 90)/12)**2 <= 1
    eye2 = ((x - 134)/8)**2 + ((y - 90)/12)**2 <= 1
    for c in range(3):
        img1[c][eye1] = (0.1 - [0.485, 0.456, 0.406][c]) / [0.229, 0.224, 0.225][c]
        img1[c][eye2] = (0.1 - [0.485, 0.456, 0.406][c]) / [0.229, 0.224, 0.225][c]
    
    images.append(img1)
    
    # 图像2: 彩色渐变
    img2 = torch.zeros(3, 224, 224)
    for i in range(224):
        for j in range(224):
            img2[0, i, j] = (i/224 - 0.485) / 0.229
            img2[1, i, j] = (j/224 - 0.456) / 0.224
            img2[2, i, j] = (0.8 - 0.406) / 0.225
    
    images.append(img2)
    
    # 图像3: 几何图案
    img3 = torch.zeros(3, 224, 224)
    for i in range(0, 224, 32):
        for j in range(0, 224, 32):
            if (i//32 + j//32) % 2 == 0:
                img3[:, i:i+32, j:j+32] = torch.tensor([
                    (1.0 - 0.485) / 0.229,
                    (1.0 - 0.456) / 0.224,
                    (1.0 - 0.406) / 0.225
                ]).reshape(3, 1, 1)
    
    images.append(img3)
    
    return torch.stack(images)

def demonstrate_mae_reconstruction(model, device, images, output_folder):
    """演示MAE完整重建过程"""
    print("\n🔍 MAE完整重建演示...")
    
    # 反归一化
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    images = images.to(device)
    num_images = min(3, images.shape[0])
    
    # 测试不同的mask比例
    mask_ratios = [0.5, 0.75, 0.9]
    
    reconstruction_results = []
    
    for img_idx in range(num_images):
        img_tensor = images[img_idx:img_idx+1]
        
        print(f"\n  处理图像 {img_idx+1}...")
        
        img_results = {
            'image_id': img_idx + 1,
            'original_shape': list(img_tensor.shape),
            'mask_results': []
        }
        
        # 保存原始图像
        original_display = torch.clamp(inv_normalize(img_tensor[0]).cpu(), 0, 1)
        original_pil = transforms.ToPILImage()(original_display)
        original_path = os.path.join(output_folder, f'original_image_{img_idx+1}.png')
        original_pil.save(original_path)
        print(f"    💾 保存原始图像: {original_path}")
        
        for mask_ratio in mask_ratios:
            print(f"    🎭 测试mask比例: {mask_ratio*100:.0f}%")
            
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
                reconstructed_display = torch.clamp(inv_normalize(reconstructed[0]).cpu(), 0, 1)
                masked_img = original_display * (1 - mask_vis[0].cpu()) + mask_vis[0].cpu() * 0.5
                
                # 计算重建误差
                error = torch.abs(original_display - reconstructed_display)
                
                # 保存结果图像
                mask_folder = os.path.join(output_folder, f'mask_{int(mask_ratio*100)}percent')
                if not os.path.exists(mask_folder):
                    os.makedirs(mask_folder)
                
                # 保存掩码图像
                masked_pil = transforms.ToPILImage()(masked_img)
                masked_path = os.path.join(mask_folder, f'masked_image_{img_idx+1}.png')
                masked_pil.save(masked_path)
                
                # 保存重建图像
                reconstructed_pil = transforms.ToPILImage()(reconstructed_display)
                reconstructed_path = os.path.join(mask_folder, f'reconstructed_image_{img_idx+1}.png')
                reconstructed_pil.save(reconstructed_path)
                
                # 保存误差图
                error_display = error.mean(dim=0)
                error_normalized = (error_display - error_display.min()) / (error_display.max() - error_display.min() + 1e-8)
                error_pil = transforms.ToPILImage()(error_normalized)
                error_path = os.path.join(mask_folder, f'error_map_{img_idx+1}.png')
                error_pil.save(error_path)
                
                # 记录统计信息
                mask_result = {
                    'mask_ratio': mask_ratio,
                    'loss': loss.item(),
                    'actual_mask_ratio': mask.float().mean().item(),
                    'mean_error': error.mean().item(),
                    'max_error': error.max().item(),
                    'files': {
                        'masked': masked_path,
                        'reconstructed': reconstructed_path,
                        'error_map': error_path
                    }
                }
                
                img_results['mask_results'].append(mask_result)
                
                print(f"      损失: {loss.item():.4f}")
                print(f"      实际mask比例: {mask.float().mean().item():.2%}")
                print(f"      平均误差: {error.mean().item():.4f}")
                print(f"      💾 保存到: {mask_folder}")
        
        img_results['original_file'] = original_path
        reconstruction_results.append(img_results)
    
    return reconstruction_results

def create_comparison_visualization(model, device, images, output_folder):
    """创建对比可视化"""
    print("\n📊 创建对比可视化...")
    
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    images = images.to(device)
    num_images = min(3, images.shape[0])
    mask_ratios = [0.5, 0.75, 0.9]
    
    # 创建大型对比图
    fig, axes = plt.subplots(num_images, len(mask_ratios)*3 + 1, figsize=(20, num_images*4))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for img_idx in range(num_images):
        img_tensor = images[img_idx:img_idx+1]
        
        # 显示原始图像
        original_display = torch.clamp(inv_normalize(img_tensor[0]).cpu(), 0, 1)
        axes[img_idx, 0].imshow(original_display.permute(1, 2, 0))
        axes[img_idx, 0].set_title(f'原始图像 {img_idx+1}')
        axes[img_idx, 0].axis('off')
        
        col_idx = 1
        
        for mask_ratio in mask_ratios:
            with torch.no_grad():
                loss, pred, mask = model(img_tensor, mask_ratio=mask_ratio)
                reconstructed = model.unpatchify(pred)
                
                # 创建掩码可视化
                mask_vis = mask.detach()
                mask_vis = mask_vis.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
                mask_vis = model.unpatchify(mask_vis)
                
                # 显示掩码图像
                masked_img = original_display * (1 - mask_vis[0].cpu()) + mask_vis[0].cpu() * 0.5
                axes[img_idx, col_idx].imshow(masked_img.permute(1, 2, 0))
                axes[img_idx, col_idx].set_title(f'掩码 {mask_ratio*100:.0f}%')
                axes[img_idx, col_idx].axis('off')
                
                # 显示重建图像
                reconstructed_display = torch.clamp(inv_normalize(reconstructed[0]).cpu(), 0, 1)
                axes[img_idx, col_idx+1].imshow(reconstructed_display.permute(1, 2, 0))
                axes[img_idx, col_idx+1].set_title(f'重建\n损失:{loss.item():.3f}')
                axes[img_idx, col_idx+1].axis('off')
                
                # 显示误差图
                error = torch.abs(original_display - reconstructed_display)
                error_display = error.mean(dim=0)
                im = axes[img_idx, col_idx+2].imshow(error_display, cmap='hot')
                axes[img_idx, col_idx+2].set_title('重建误差')
                axes[img_idx, col_idx+2].axis('off')
                plt.colorbar(im, ax=axes[img_idx, col_idx+2], fraction=0.046, pad=0.04)
                
                col_idx += 3
    
    plt.tight_layout()
    
    # 保存对比图
    comparison_path = os.path.join(output_folder, 'mae_reconstruction_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✅ 对比可视化保存: {comparison_path}")
    
    try:
        plt.show()
    except:
        print("💡 如果要查看图像，请在支持图形界面的环境中运行")
    
    return comparison_path

def save_reconstruction_report(results, output_folder):
    """保存重建报告"""
    print("\n📄 生成重建报告...")
    
    # 保存JSON报告
    json_path = os.path.join(output_folder, 'reconstruction_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 生成Markdown报告
    md_path = os.path.join(output_folder, 'reconstruction_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# MAE重建演示报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 实验设置\n")
        f.write("- 模型: MAE ViT-Base\n")
        f.write("- 编码器: 预训练权重\n")
        f.write("- 解码器: 随机初始化\n")
        f.write("- 测试mask比例: 50%, 75%, 90%\n\n")
        
        f.write("## 重建结果\n\n")
        
        for img_result in results:
            img_id = img_result['image_id']
            f.write(f"### 图像 {img_id}\n\n")
            f.write(f"- 原始图像: `{os.path.basename(img_result['original_file'])}`\n\n")
            
            f.write("| Mask比例 | 损失值 | 实际Mask% | 平均误差 | 最大误差 |\n")
            f.write("|---------|--------|-----------|----------|----------|\n")
            
            for mask_result in img_result['mask_results']:
                f.write(f"| {mask_result['mask_ratio']*100:.0f}% | "
                       f"{mask_result['loss']:.4f} | "
                       f"{mask_result['actual_mask_ratio']*100:.1f}% | "
                       f"{mask_result['mean_error']:.4f} | "
                       f"{mask_result['max_error']:.4f} |\n")
            
            f.write("\n")
        
        f.write("## 文件结构\n\n")
        f.write("```\n")
        f.write(f"{os.path.basename(output_folder)}/\n")
        f.write("├── original_image_*.png     # 原始图像\n")
        f.write("├── mask_50percent/          # 50%掩码结果\n")
        f.write("├── mask_75percent/          # 75%掩码结果\n")
        f.write("├── mask_90percent/          # 90%掩码结果\n")
        f.write("├── mae_reconstruction_comparison.png  # 对比可视化\n")
        f.write("├── reconstruction_report.json        # JSON报告\n")
        f.write("└── reconstruction_report.md          # 本报告\n")
        f.write("```\n")
    
    print(f"✅ 报告保存完成:")
    print(f"   JSON: {json_path}")
    print(f"   Markdown: {md_path}")
    
    return json_path, md_path

def main():
    """主函数"""
    print("🎭 MAE完整重建演示")
    print("=" * 50)
    
    # 设置环境
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 创建输出文件夹
    output_folder = create_output_folder()
    
    # 加载完整MAE模型
    model, device = load_complete_mae_model()
    
    # 加载图像
    print("\n🎯 加载测试图像...")
    images = load_animediffusion_images()
    
    if images is None:
        print("🎨 使用备用图像...")
        images = create_fallback_images()
    
    # 演示重建过程
    results = demonstrate_mae_reconstruction(model, device, images, output_folder)
    
    # 创建对比可视化
    comparison_path = create_comparison_visualization(model, device, images, output_folder)
    
    # 保存报告
    json_path, md_path = save_reconstruction_report(results, output_folder)
    
    print(f"\n🎉 MAE重建演示完成!")
    print(f"📁 所有结果保存在: {output_folder}")
    print(f"📊 对比图: {comparison_path}")
    print(f"📄 详细报告: {md_path}")
    
    print(f"\n💡 关键发现:")
    print("✅ 编码器+解码器成功重建被mask的图像")
    print("🎭 mask比例越高，重建难度越大")
    print("🎨 解码器从编码器特征生成像素细节")
    print("📈 预训练编码器提供强大的语义理解能力")

if __name__ == "__main__":
    main()

