#!/usr/bin/env python3
"""
诊断MAE重建噪声问题
分析可能的原因：模型、解码器、数据预处理等
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_mae_model_variants():
    """加载不同配置的MAE模型进行对比"""
    print("\n🔍 加载不同MAE模型配置...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"设备: {device}")
    
    models = {}
    
    # 1. 完全随机初始化的模型
    print("  📦 创建随机初始化模型...")
    random_model = models_mae.mae_vit_base_patch16()
    random_model = random_model.to(device)
    random_model.eval()
    models['random'] = random_model
    
    # 2. 只加载编码器预训练权重的模型
    print("  📦 创建编码器预训练模型...")
    encoder_pretrained = models_mae.mae_vit_base_patch16()
    
    pretrain_path = 'pretrained_models/mae_pretrain_vit_base.pth'
    if os.path.exists(pretrain_path):
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        
        # 只加载编码器权重
        encoder_state_dict = {}
        for key, value in checkpoint['model'].items():
            if not key.startswith('decoder') and key != 'mask_token':
                encoder_state_dict[key] = value
        
        encoder_pretrained.load_state_dict(encoder_state_dict, strict=False)
        print("    ✅ 编码器预训练权重加载成功")
    
    encoder_pretrained = encoder_pretrained.to(device)
    encoder_pretrained.eval()
    models['encoder_pretrained'] = encoder_pretrained
    
    # 3. 检查是否有完整的预训练模型（包含解码器）
    print("  📦 检查完整预训练模型...")
    try:
        # 尝试加载可能包含解码器的模型
        full_model = models_mae.mae_vit_base_patch16()
        
        # 检查checkpoint中的keys
        if os.path.exists(pretrain_path):
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            decoder_keys = [k for k in checkpoint['model'].keys() if k.startswith('decoder')]
            
            if len(decoder_keys) > 0:
                print(f"    ✅ 发现解码器权重: {len(decoder_keys)} 个参数")
                full_model.load_state_dict(checkpoint['model'])
                full_model = full_model.to(device)
                full_model.eval()
                models['full_pretrained'] = full_model
            else:
                print("    ❌ 预训练模型中没有解码器权重")
        
    except Exception as e:
        print(f"    ❌ 加载完整模型失败: {e}")
    
    return models, device

def analyze_model_components(models, device):
    """分析模型各组件的状态"""
    print("\n🔬 分析模型组件...")
    
    for model_name, model in models.items():
        print(f"\n  📊 {model_name} 模型分析:")
        
        # 检查编码器参数统计
        encoder_params = []
        for name, param in model.named_parameters():
            if not name.startswith('decoder') and name != 'mask_token':
                encoder_params.append(param.data.flatten())
        
        if encoder_params:
            encoder_tensor = torch.cat(encoder_params)
            print(f"    编码器参数统计:")
            print(f"      均值: {encoder_tensor.mean().item():.6f}")
            print(f"      标准差: {encoder_tensor.std().item():.6f}")
            print(f"      范围: [{encoder_tensor.min().item():.6f}, {encoder_tensor.max().item():.6f}]")
        
        # 检查解码器参数统计
        decoder_params = []
        for name, param in model.named_parameters():
            if name.startswith('decoder') or name == 'mask_token':
                decoder_params.append(param.data.flatten())
        
        if decoder_params:
            decoder_tensor = torch.cat(decoder_params)
            print(f"    解码器参数统计:")
            print(f"      均值: {decoder_tensor.mean().item():.6f}")
            print(f"      标准差: {decoder_tensor.std().item():.6f}")
            print(f"      范围: [{decoder_tensor.min().item():.6f}, {decoder_tensor.max().item():.6f}]")
        else:
            print("    ❌ 没有解码器参数")

def test_reconstruction_quality(models, device):
    """测试不同模型的重建质量"""
    print("\n🎯 测试重建质量...")
    
    # 创建简单测试图像
    test_img = create_simple_test_image(device)
    
    # 反归一化用于显示
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n  🔍 测试 {model_name} 模型...")
        
        with torch.no_grad():
            try:
                # 测试重建
                loss, pred, mask = model(test_img, mask_ratio=0.75)
                reconstructed = model.unpatchify(pred)
                
                # 计算统计信息
                pred_stats = {
                    'mean': pred.mean().item(),
                    'std': pred.std().item(),
                    'min': pred.min().item(),
                    'max': pred.max().item()
                }
                
                reconstructed_stats = {
                    'mean': reconstructed.mean().item(),
                    'std': reconstructed.std().item(),
                    'min': reconstructed.min().item(),
                    'max': reconstructed.max().item()
                }
                
                results[model_name] = {
                    'loss': loss.item(),
                    'pred_stats': pred_stats,
                    'reconstructed_stats': reconstructed_stats,
                    'success': True
                }
                
                print(f"    损失: {loss.item():.4f}")
                print(f"    预测统计: 均值={pred_stats['mean']:.4f}, 标准差={pred_stats['std']:.4f}")
                print(f"    重建统计: 均值={reconstructed_stats['mean']:.4f}, 标准差={reconstructed_stats['std']:.4f}")
                
                # 检查是否有异常值
                if abs(pred_stats['mean']) > 10 or pred_stats['std'] > 10:
                    print(f"    ⚠️  预测值异常！可能存在梯度爆炸或参数初始化问题")
                
                if abs(reconstructed_stats['mean']) > 5 or reconstructed_stats['std'] > 5:
                    print(f"    ⚠️  重建值异常！可能导致噪声")
                
            except Exception as e:
                print(f"    ❌ 测试失败: {e}")
                results[model_name] = {'success': False, 'error': str(e)}
    
    return results

def create_simple_test_image(device):
    """创建简单的测试图像"""
    # 创建一个简单的渐变图像
    img = torch.zeros(1, 3, 224, 224, device=device)
    
    for i in range(224):
        for j in range(224):
            # 标准化后的渐变
            r = (i / 224 - 0.485) / 0.229
            g = (j / 224 - 0.456) / 0.224
            b = (0.5 - 0.406) / 0.225
            
            img[0, 0, i, j] = r
            img[0, 1, i, j] = g
            img[0, 2, i, j] = b
    
    return img

def diagnose_noise_sources(models, device):
    """诊断噪声来源"""
    print("\n🔍 诊断噪声来源...")
    
    # 创建测试图像
    test_img = create_simple_test_image(device)
    
    # 反归一化
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    fig, axes = plt.subplots(len(models), 5, figsize=(20, len(models)*4))
    
    if len(models) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (model_name, model) in enumerate(models.items()):
        print(f"\n  🔬 分析 {model_name}...")
        
        with torch.no_grad():
            try:
                # 1. 原始图像
                original_display = torch.clamp(inv_normalize(test_img[0]).cpu(), 0, 1)
                axes[i, 0].imshow(original_display.permute(1, 2, 0))
                axes[i, 0].set_title(f'{model_name}\n原始图像')
                axes[i, 0].axis('off')
                
                # 2. MAE前向传播
                loss, pred, mask = model(test_img, mask_ratio=0.75)
                
                # 3. 显示预测的原始值（未unpatchify）
                pred_vis = pred[0, :16].cpu().numpy()  # 取前16个patch的预测
                pred_2d = pred_vis.reshape(4, 4)
                im1 = axes[i, 1].imshow(pred_2d, cmap='RdBu_r')
                axes[i, 1].set_title(f'预测原始值\n范围:[{pred.min():.2f}, {pred.max():.2f}]')
                axes[i, 1].axis('off')
                plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
                
                # 4. 重建图像
                reconstructed = model.unpatchify(pred)
                reconstructed_display = torch.clamp(inv_normalize(reconstructed[0]).cpu(), 0, 1)
                axes[i, 2].imshow(reconstructed_display.permute(1, 2, 0))
                axes[i, 2].set_title(f'重建图像\n损失:{loss.item():.3f}')
                axes[i, 2].axis('off')
                
                # 5. 重建误差
                error = torch.abs(original_display - reconstructed_display)
                error_display = error.mean(dim=0)
                im2 = axes[i, 3].imshow(error_display, cmap='hot')
                axes[i, 3].set_title(f'重建误差\n均值:{error.mean():.3f}')
                axes[i, 3].axis('off')
                plt.colorbar(im2, ax=axes[i, 3], fraction=0.046, pad=0.04)
                
                # 6. 诊断信息
                axes[i, 4].text(0.1, 0.9, f'模型: {model_name}', transform=axes[i, 4].transAxes, fontsize=10, weight='bold')
                axes[i, 4].text(0.1, 0.8, f'损失: {loss.item():.4f}', transform=axes[i, 4].transAxes, fontsize=9)
                axes[i, 4].text(0.1, 0.7, f'预测范围: [{pred.min():.2f}, {pred.max():.2f}]', transform=axes[i, 4].transAxes, fontsize=9)
                axes[i, 4].text(0.1, 0.6, f'预测均值: {pred.mean():.4f}', transform=axes[i, 4].transAxes, fontsize=9)
                axes[i, 4].text(0.1, 0.5, f'预测标准差: {pred.std():.4f}', transform=axes[i, 4].transAxes, fontsize=9)
                
                # 诊断结果
                if abs(pred.mean().item()) > 1.0:
                    axes[i, 4].text(0.1, 0.3, '⚠️ 预测均值异常', transform=axes[i, 4].transAxes, fontsize=9, color='red')
                
                if pred.std().item() > 2.0:
                    axes[i, 4].text(0.1, 0.2, '⚠️ 预测方差过大', transform=axes[i, 4].transAxes, fontsize=9, color='red')
                
                if error.mean().item() > 0.5:
                    axes[i, 4].text(0.1, 0.1, '⚠️ 重建误差过大', transform=axes[i, 4].transAxes, fontsize=9, color='red')
                
                axes[i, 4].set_xlim(0, 1)
                axes[i, 4].set_ylim(0, 1)
                axes[i, 4].axis('off')
                axes[i, 4].set_title('诊断信息')
                
            except Exception as e:
                axes[i, 0].text(0.5, 0.5, f'错误: {str(e)}', ha='center', va='center', transform=axes[i, 0].transAxes)
                for j in range(5):
                    axes[i, j].axis('off')
    
    plt.tight_layout()
    
    # 保存诊断结果
    diagnosis_path = 'mae_noise_diagnosis.png'
    plt.savefig(diagnosis_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 诊断结果保存: {diagnosis_path}")
    
    try:
        plt.show()
    except:
        print("💡 如果要查看图像，请在支持图形界面的环境中运行")
    
    return diagnosis_path

def provide_solutions():
    """提供解决方案"""
    print("\n💡 噪声问题解决方案:")
    print("=" * 50)
    
    print("🔍 可能的原因:")
    print("1. 解码器未预训练 - 随机初始化导致输出不稳定")
    print("2. 预测值范围异常 - 可能需要调整损失函数或归一化")
    print("3. 模型架构不匹配 - 编码器和解码器版本不兼容")
    print("4. 数据预处理问题 - 归一化参数不正确")
    print()
    
    print("🛠️ 解决方案:")
    print("1. 使用完整预训练模型:")
    print("   - 下载包含解码器权重的完整MAE模型")
    print("   - 或者在自己的数据上训练解码器")
    print()
    
    print("2. 调整解码器初始化:")
    print("   - 使用更小的初始化权重")
    print("   - 添加权重正则化")
    print()
    
    print("3. 修改损失函数:")
    print("   - 使用更稳定的重建损失")
    print("   - 添加平滑正则项")
    print()
    
    print("4. 数据预处理优化:")
    print("   - 检查归一化参数")
    print("   - 使用更适合的数据范围")

def main():
    """主函数"""
    print("🔍 MAE重建噪声诊断")
    print("=" * 50)
    
    # 设置环境
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 加载不同配置的模型
    models, device = load_mae_model_variants()
    
    # 分析模型组件
    analyze_model_components(models, device)
    
    # 测试重建质量
    test_results = test_reconstruction_quality(models, device)
    
    # 诊断噪声来源
    diagnosis_path = diagnose_noise_sources(models, device)
    
    # 提供解决方案
    provide_solutions()
    
    print(f"\n🎯 诊断结论:")
    print("根据你看到的噪声图像，最可能的原因是:")
    print("❌ 解码器使用随机初始化权重，没有经过预训练")
    print("🎨 编码器虽然预训练了，但解码器不知道如何正确重建像素")
    print("💡 这就是为什么重建结果是噪声而不是清晰图像")
    
    print(f"\n📁 诊断结果: {diagnosis_path}")

if __name__ == "__main__":
    main()

