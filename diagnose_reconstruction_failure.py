#!/usr/bin/env python3
"""
诊断MAE重建完全失败的问题
分析为什么连未掩盖的像素都无法正确显示
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

# 解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import models_mae

def diagnose_reconstruction_pipeline():
    """诊断重建流程的每个步骤"""
    print("🔍 诊断MAE重建流程...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 加载模型
    model = models_mae.mae_vit_base_patch16(norm_pix_loss=True)
    
    # 尝试加载训练好的模型
    checkpoint_path = './output_image_repair_v1/checkpoint-19.pth'
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'])
            print("✅ 加载训练模型")
        except Exception as e:
            print(f"⚠️  加载失败，使用随机模型: {e}")
    else:
        print("⚠️  使用随机初始化模型")
    
    model.to(device)
    model.eval()
    
    # 加载测试图片
    try:
        ds = load_dataset("Mercity/AnimeDiffusion_Dataset")
        sample = ds['train'][0]
        original_img = sample['image']
        
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        
        print(f"✅ 原始图片: {original_img.size}, 模式: {original_img.mode}")
    except Exception as e:
        print(f"❌ 图片加载失败: {e}")
        return
    
    # 步骤1: 检查图像预处理
    print(f"\n🔍 步骤1: 检查图像预处理...")
    
    transform = transforms.Compose([
        transforms.Resize(int(224 * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 只做resize和crop，不做归一化
    transform_no_norm = transforms.Compose([
        transforms.Resize(int(224 * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    img_tensor = transform(original_img).unsqueeze(0).to(device)
    img_tensor_no_norm = transform_no_norm(original_img).unsqueeze(0).to(device)
    
    print(f"  预处理后形状: {img_tensor.shape}")
    print(f"  归一化后范围: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    print(f"  未归一化范围: [{img_tensor_no_norm.min():.3f}, {img_tensor_no_norm.max():.3f}]")
    
    # 步骤2: 检查模型前向传播
    print(f"\n🔍 步骤2: 检查模型前向传播...")
    
    with torch.no_grad():
        # 获取中间结果
        x = model.patch_embed(img_tensor)
        print(f"  Patch embedding形状: {x.shape}")
        print(f"  Patch embedding范围: [{x.min():.3f}, {x.max():.3f}]")
        
        # 添加位置编码
        x = x + model.pos_embed[:, 1:, :]
        print(f"  添加位置编码后范围: [{x.min():.3f}, {x.max():.3f}]")
        
        # 执行完整的前向传播
        loss, pred, mask = model(img_tensor, mask_ratio=0.25)
        print(f"  预测形状: {pred.shape}")
        print(f"  预测范围: [{pred.min():.3f}, {pred.max():.3f}]")
        print(f"  掩码形状: {mask.shape}")
        print(f"  掩码值: {mask.unique()}")
    
    # 步骤3: 检查unpatchify过程
    print(f"\n🔍 步骤3: 检查unpatchify过程...")
    
    reconstructed = model.unpatchify(pred)
    print(f"  重建图像形状: {reconstructed.shape}")
    print(f"  重建图像范围: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    
    # 步骤4: 检查反归一化
    print(f"\n🔍 步骤4: 检查反归一化...")
    
    # 标准反归一化
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    original_display = torch.clamp(inv_normalize(img_tensor[0]).cpu(), 0, 1)
    recon_display = torch.clamp(inv_normalize(reconstructed[0]).cpu(), 0, 1)
    
    print(f"  原图反归一化范围: [{original_display.min():.3f}, {original_display.max():.3f}]")
    print(f"  重建反归一化范围: [{recon_display.min():.3f}, {recon_display.max():.3f}]")
    
    # 步骤5: 可视化诊断
    print(f"\n🔍 步骤5: 创建诊断可视化...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 第一行：处理流程
    axes[0, 0].imshow(original_img.resize((224, 224)))
    axes[0, 0].set_title('Original (resized)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_tensor_no_norm[0].permute(1, 2, 0).cpu())
    axes[0, 1].set_title('After Transform\n(no normalization)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(original_display.permute(1, 2, 0))
    axes[0, 2].set_title('After Normalization\n(inv-normalized)')
    axes[0, 2].axis('off')
    
    # 显示预测的原始值（不做clamp）
    recon_raw = inv_normalize(reconstructed[0]).cpu()
    axes[0, 3].imshow(recon_raw.permute(1, 2, 0))
    axes[0, 3].set_title('Raw Reconstruction\n(no clamp)')
    axes[0, 3].axis('off')
    
    # 第二行：问题分析
    axes[1, 0].imshow(recon_display.permute(1, 2, 0))
    axes[1, 0].set_title(f'Clamped Reconstruction\nLoss: {loss.item():.3f}')
    axes[1, 0].axis('off')
    
    # 显示预测值的分布
    pred_flat = pred.flatten().cpu().numpy()
    axes[1, 1].hist(pred_flat, bins=50, alpha=0.7)
    axes[1, 1].set_title('Prediction Distribution')
    axes[1, 1].set_xlabel('Prediction Value')
    axes[1, 1].set_ylabel('Frequency')
    
    # 显示重建值的分布
    recon_flat = reconstructed.flatten().cpu().numpy()
    axes[1, 2].hist(recon_flat, bins=50, alpha=0.7, color='orange')
    axes[1, 2].set_title('Reconstruction Distribution')
    axes[1, 2].set_xlabel('Pixel Value')
    axes[1, 2].set_ylabel('Frequency')
    
    # 显示掩码
    mask_vis = mask.detach().unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
    mask_vis = model.unpatchify(mask_vis)
    axes[1, 3].imshow(mask_vis[0].cpu().permute(1, 2, 0), cmap='gray')
    axes[1, 3].set_title(f'Mask Visualization\n{mask.float().mean().item():.1%} masked')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('reconstruction_diagnosis.png', dpi=150, bbox_inches='tight')
    print("✅ 诊断结果保存: reconstruction_diagnosis.png")
    plt.close()
    
    return {
        'loss': loss.item(),
        'pred_range': (pred.min().item(), pred.max().item()),
        'recon_range': (reconstructed.min().item(), reconstructed.max().item()),
        'mask_ratio': mask.float().mean().item()
    }

def test_simple_reconstruction():
    """测试简单的重建流程"""
    print(f"\n🧪 测试简化的重建流程...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 创建一个简单的测试图像
    test_img = torch.ones(1, 3, 224, 224, device=device) * 0.5  # 灰色图像
    
    # 添加一些简单的图案
    test_img[0, 0, 50:100, 50:100] = 1.0  # 红色方块
    test_img[0, 1, 150:200, 150:200] = 1.0  # 绿色方块
    test_img[0, 2, 100:150, 100:150] = 1.0  # 蓝色方块
    
    print(f"  测试图像范围: [{test_img.min():.3f}, {test_img.max():.3f}]")
    
    # 测试不同的模型
    models_to_test = [
        ('随机初始化', None),
        ('训练模型', './output_image_repair_v1/checkpoint-19.pth')
    ]
    
    fig, axes = plt.subplots(len(models_to_test), 4, figsize=(16, len(models_to_test)*4))
    
    for i, (model_name, checkpoint_path) in enumerate(models_to_test):
        print(f"\n  测试 {model_name}...")
        
        model = models_mae.mae_vit_base_patch16(norm_pix_loss=True)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['model'])
                print(f"    ✅ 模型加载成功")
            except Exception as e:
                print(f"    ⚠️  模型加载失败: {e}")
        
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            loss, pred, mask = model(test_img, mask_ratio=0.25)
            reconstructed = model.unpatchify(pred)
            
            # 创建掩码可视化
            mask_vis = mask.detach().unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
            mask_vis = model.unpatchify(mask_vis)
        
        # 显示结果
        masked_img = test_img[0].cpu() * (1 - mask_vis[0].cpu()) + mask_vis[0].cpu() * 0.5
        
        axes[i, 0].imshow(test_img[0].cpu().permute(1, 2, 0))
        axes[i, 0].set_title(f'{model_name}\nOriginal Test Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(masked_img.permute(1, 2, 0))
        axes[i, 1].set_title('25% Masked')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(torch.clamp(reconstructed[0].cpu(), 0, 1).permute(1, 2, 0))
        axes[i, 2].set_title(f'Reconstructed\nLoss: {loss.item():.3f}')
        axes[i, 2].axis('off')
        
        # 显示预测值的统计
        axes[i, 3].text(0.1, 0.8, f'Prediction Stats:', transform=axes[i, 3].transAxes, fontweight='bold')
        axes[i, 3].text(0.1, 0.7, f'Min: {pred.min():.3f}', transform=axes[i, 3].transAxes)
        axes[i, 3].text(0.1, 0.6, f'Max: {pred.max():.3f}', transform=axes[i, 3].transAxes)
        axes[i, 3].text(0.1, 0.5, f'Mean: {pred.mean():.3f}', transform=axes[i, 3].transAxes)
        axes[i, 3].text(0.1, 0.4, f'Std: {pred.std():.3f}', transform=axes[i, 3].transAxes)
        axes[i, 3].text(0.1, 0.2, f'Recon Range:', transform=axes[i, 3].transAxes, fontweight='bold')
        axes[i, 3].text(0.1, 0.1, f'[{reconstructed.min():.3f}, {reconstructed.max():.3f}]', transform=axes[i, 3].transAxes)
        axes[i, 3].axis('off')
        
        print(f"    损失: {loss.item():.4f}")
        print(f"    预测范围: [{pred.min():.3f}, {pred.max():.3f}]")
        print(f"    重建范围: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    
    plt.tight_layout()
    plt.savefig('simple_reconstruction_test.png', dpi=150, bbox_inches='tight')
    print("✅ 简单重建测试保存: simple_reconstruction_test.png")
    plt.close()

def check_model_components():
    """检查模型各个组件的状态"""
    print(f"\n🔍 检查模型组件状态...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = models_mae.mae_vit_base_patch16(norm_pix_loss=True)
    
    # 加载训练模型
    checkpoint_path = './output_image_repair_v1/checkpoint-19.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
    
    model.to(device)
    model.eval()
    
    # 检查关键组件的权重
    print(f"📊 模型组件检查:")
    
    # 检查patch embedding
    patch_embed_weight = model.patch_embed.proj.weight
    print(f"  Patch embedding权重范围: [{patch_embed_weight.min():.3f}, {patch_embed_weight.max():.3f}]")
    print(f"  Patch embedding权重标准差: {patch_embed_weight.std():.3f}")
    
    # 检查decoder prediction层
    decoder_pred_weight = model.decoder_pred.weight
    decoder_pred_bias = model.decoder_pred.bias
    print(f"  Decoder prediction权重范围: [{decoder_pred_weight.min():.3f}, {decoder_pred_weight.max():.3f}]")
    print(f"  Decoder prediction偏置范围: [{decoder_pred_bias.min():.3f}, {decoder_pred_bias.max():.3f}]")
    
    # 检查位置编码
    pos_embed = model.pos_embed
    print(f"  位置编码范围: [{pos_embed.min():.3f}, {pos_embed.max():.3f}]")
    
    # 测试一个简单的前向传播
    test_input = torch.randn(1, 3, 224, 224, device=device)
    
    with torch.no_grad():
        loss, pred, mask = model(test_input, mask_ratio=0.25)
        
        print(f"\n🧪 简单测试结果:")
        print(f"  输入范围: [{test_input.min():.3f}, {test_input.max():.3f}]")
        print(f"  损失: {loss.item():.4f}")
        print(f"  预测范围: [{pred.min():.3f}, {pred.max():.3f}]")
        
        # 检查是否有异常值
        if torch.isnan(pred).any():
            print("  ❌ 预测中包含NaN值!")
        if torch.isinf(pred).any():
            print("  ❌ 预测中包含无穷值!")
        
        # 检查梯度
        if hasattr(model, 'decoder_pred'):
            if model.decoder_pred.weight.grad is not None:
                grad_norm = model.decoder_pred.weight.grad.norm()
                print(f"  梯度范数: {grad_norm:.6f}")

def identify_core_issues():
    """识别核心问题"""
    print(f"\n🎯 核心问题分析:")
    
    issues = [
        {
            'issue': 'norm_pix_loss 设置问题',
            'description': 'norm_pix_loss=True 会改变目标的计算方式',
            'solution': '尝试 norm_pix_loss=False',
            'severity': 'HIGH'
        },
        {
            'issue': '训练目标不匹配',
            'description': '模型可能学习的是归一化像素而不是原始像素',
            'solution': '检查损失函数的计算',
            'severity': 'HIGH'
        },
        {
            'issue': '反归一化错误',
            'description': '重建结果的反归一化可能不正确',
            'solution': '验证归一化/反归一化的一致性',
            'severity': 'MEDIUM'
        },
        {
            'issue': '模型未充分收敛',
            'description': '即使训练20个epoch，模型可能仍未学会正确重建',
            'solution': '大幅增加训练时间或降低学习率',
            'severity': 'MEDIUM'
        }
    ]
    
    severity_colors = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}
    
    for i, issue in enumerate(issues, 1):
        print(f"\n{i}. {severity_colors[issue['severity']]} {issue['issue']}")
        print(f"   问题: {issue['description']}")
        print(f"   解决: {issue['solution']}")

def create_fix_attempt():
    """创建修复尝试"""
    print(f"\n🛠️ 创建修复尝试配置...")
    
    # 最可能的修复方案
    fix_config = """# 修复重建问题的尝试
python main_pretrain_animediffusion.py \\
    --mask_ratio 0.25 \\
    --epochs 50 \\
    --batch_size 4 \\
    --accum_iter 16 \\
    --blr 1e-5 \\
    --warmup_epochs 15 \\
    --max_samples 1000 \\
    --weight_decay 0.01 \\
    --output_dir ./output_fix_attempt \\
    --log_dir ./output_fix_attempt \\
    --norm_pix_loss  # 尝试关闭这个选项
"""
    
    print(fix_config)
    
    # 保存修复配置
    with open('fix_reconstruction.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("export KMP_DUPLICATE_LIB_OK=TRUE\n\n")
        f.write("# 尝试修复重建问题\n")
        f.write("echo '🛠️  尝试修复MAE重建问题...'\n\n")
        f.write(fix_config.replace('\\', '\\'))
    
    os.chmod('fix_reconstruction.sh', 0o755)
    print(f"✅ 修复配置保存: fix_reconstruction.sh")

def main():
    """主函数"""
    print("🔍 MAE重建失败诊断工具")
    print("=" * 50)
    
    # 1. 诊断重建流程
    diagnosis_result = diagnose_reconstruction_pipeline()
    
    # 2. 测试简单重建
    test_simple_reconstruction()
    
    # 3. 检查模型组件
    check_model_components()
    
    # 4. 识别核心问题
    identify_core_issues()
    
    # 5. 创建修复尝试
    create_fix_attempt()
    
    print(f"\n💡 最可能的问题:")
    print(f"  🔴 norm_pix_loss=True 可能导致目标计算错误")
    print(f"  🔴 模型输出值范围异常")
    print(f"  🔴 训练时间仍然不足")
    
    print(f"\n🎯 立即尝试的解决方案:")
    print(f"  1. 关闭 norm_pix_loss")
    print(f"  2. 使用更低的学习率")
    print(f"  3. 增加训练时间")

if __name__ == "__main__":
    main()


