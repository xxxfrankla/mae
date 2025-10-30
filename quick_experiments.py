#!/usr/bin/env python3
"""
MAE 快速实验脚本
在 Apple M4 上进行各种 MAE 实验
"""

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import models_mae
import models_vit

def load_and_test_pretrained():
    """加载并测试预训练模型"""
    print("🔄 测试预训练模型加载...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 检查模型文件
    pretrain_path = 'pretrained_models/mae_pretrain_vit_base.pth'
    finetune_path = 'pretrained_models/mae_finetuned_vit_base.pth'
    
    if os.path.exists(pretrain_path):
        print(f"✅ 找到预训练模型: {pretrain_path}")
        
        # 加载 MAE 预训练模型
        model = models_mae.mae_vit_base_patch16()
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        model.eval()
        
        print("✅ MAE 预训练模型加载成功")
        
        # 测试推理
        x = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            loss, pred, mask = model(x, mask_ratio=0.75)
        
        print(f"  损失值: {loss.item():.4f}")
        print(f"  掩码比例: {mask.float().mean().item():.2f}")
        
    else:
        print(f"❌ 未找到预训练模型: {pretrain_path}")
        print("请先运行: ./download_models.sh")
    
    if os.path.exists(finetune_path):
        print(f"✅ 找到微调模型: {finetune_path}")
        
        # 加载分类模型
        model = models_vit.vit_base_patch16(num_classes=1000)
        checkpoint = torch.load(finetune_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        model.eval()
        
        print("✅ 分类模型加载成功")
        
        # 测试分类
        x = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            logits = model(x)
        
        pred_class = logits.argmax(dim=1).item()
        confidence = F.softmax(logits, dim=1).max().item()
        
        print(f"  预测类别: {pred_class}")
        print(f"  置信度: {confidence:.4f}")
        
    else:
        print(f"❌ 未找到微调模型: {finetune_path}")

def create_sample_image():
    """创建测试图像"""
    print("\n🎨 创建测试图像...")
    
    # 创建一个简单的测试图像
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # 添加一些几何图形
    # 红色圆形
    center = (112, 112)
    radius = 50
    y, x = np.ogrid[:224, :224]
    mask_circle = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    img[mask_circle] = [255, 0, 0]
    
    # 蓝色矩形
    img[50:100, 150:200] = [0, 0, 255]
    
    # 绿色三角形
    for i in range(50):
        img[150+i, 50:50+i] = [0, 255, 0]
    
    # 保存图像
    pil_img = Image.fromarray(img)
    pil_img.save('test_image.png')
    
    print("✅ 测试图像已保存: test_image.png")
    return pil_img

def test_mae_reconstruction():
    """测试 MAE 重建功能"""
    print("\n🔄 测试 MAE 图像重建...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 创建测试图像
    test_img = create_sample_image()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(test_img).unsqueeze(0).to(device)
    
    # 加载模型
    model = models_mae.mae_vit_base_patch16()
    model = model.to(device)
    model.eval()
    
    # 进行重建
    with torch.no_grad():
        loss, pred, mask = model(img_tensor, mask_ratio=0.75)
    
    print(f"✅ 重建完成")
    print(f"  重建损失: {loss.item():.4f}")
    print(f"  掩码数量: {mask.sum().item()}/{mask.numel()}")
    
    # 可视化结果（简化版）
    print("💡 提示: 运行 jupyter notebook 查看完整的可视化效果")

def benchmark_performance():
    """性能基准测试"""
    print("\n⚡ 性能基准测试...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = models_mae.mae_vit_base_patch16().to(device)
    model.eval()
    
    batch_sizes = [1, 2, 4, 8]
    
    for bs in batch_sizes:
        try:
            x = torch.randn(bs, 3, 224, 224, device=device)
            
            # 预热
            with torch.no_grad():
                _ = model(x, mask_ratio=0.75)
            
            # 计时
            import time
            start_time = time.time()
            
            num_runs = 10
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = model(x, mask_ratio=0.75)
            
            if device.type == 'mps':
                torch.mps.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs
            
            print(f"  Batch size {bs}: {avg_time*1000:.1f}ms/batch ({avg_time*1000/bs:.1f}ms/image)")
            
            # 清理内存
            del x
            if device.type == 'mps':
                torch.mps.empty_cache()
                
        except Exception as e:
            print(f"  Batch size {bs}: 内存不足 - {e}")
            break

def main():
    """主函数"""
    print("🍎 MAE Apple M4 快速实验")
    print("=" * 50)
    
    # 1. 测试预训练模型
    load_and_test_pretrained()
    
    # 2. 测试重建功能
    test_mae_reconstruction()
    
    # 3. 性能测试
    benchmark_performance()
    
    print("\n🎉 实验完成！")
    print("\n📝 下一步建议:")
    print("1. 运行 jupyter notebook demo/mae_visualize.ipynb 查看可视化")
    print("2. 下载预训练模型: ./download_models.sh")
    print("3. 准备自己的数据集进行微调实验")

if __name__ == "__main__":
    main()
