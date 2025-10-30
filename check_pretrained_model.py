#!/usr/bin/env python3
"""
检查预训练MAE模型的脚本
不需要复杂依赖，只检查模型是否能正确加载
"""

import os
import sys

def check_files():
    """检查必要的文件是否存在"""
    print("🔍 检查文件...")
    
    # 检查模型文件
    pretrain_path = 'pretrained_models/mae_pretrain_vit_base.pth'
    finetune_path = 'pretrained_models/mae_finetuned_vit_base.pth'
    
    files_to_check = [
        ('预训练模型', pretrain_path),
        ('微调模型', finetune_path),
        ('MAE模型定义', 'models_mae.py'),
        ('ViT模型定义', 'models_vit.py'),
    ]
    
    all_exist = True
    for name, path in files_to_check:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)  # MB
            print(f"  ✅ {name}: {path} ({size:.1f} MB)")
        else:
            print(f"  ❌ {name}: {path} (不存在)")
            all_exist = False
    
    return all_exist

def check_python_env():
    """检查Python环境"""
    print("\n🐍 检查Python环境...")
    print(f"  Python版本: {sys.version}")
    
    # 检查必要的包
    packages = ['torch', 'torchvision', 'numpy', 'PIL', 'matplotlib']
    missing_packages = []
    
    for pkg in packages:
        try:
            if pkg == 'PIL':
                import PIL
                print(f"  ✅ {pkg}: {PIL.__version__}")
            else:
                module = __import__(pkg)
                version = getattr(module, '__version__', '未知版本')
                print(f"  ✅ {pkg}: {version}")
        except ImportError:
            print(f"  ❌ {pkg}: 未安装")
            missing_packages.append(pkg)
    
    if missing_packages:
        print(f"\n⚠️  缺少包: {', '.join(missing_packages)}")
        print("安装命令:")
        if 'torch' in missing_packages:
            print("  pip install torch torchvision")
        if 'numpy' in missing_packages:
            print("  pip install numpy")
        if 'PIL' in missing_packages:
            print("  pip install Pillow")
        if 'matplotlib' in missing_packages:
            print("  pip install matplotlib")
        return False
    
    return True

def test_model_loading():
    """测试模型加载"""
    print("\n🤖 测试模型加载...")
    
    try:
        import torch
        import models_mae
        
        # 检查设备
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"  ✅ 设备: Apple Silicon MPS")
        else:
            device = torch.device('cpu')
            print(f"  ✅ 设备: CPU")
        
        # 创建模型
        print("  🔄 创建MAE模型...")
        model = models_mae.mae_vit_base_patch16()
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✅ 模型创建成功，参数量: {total_params:,} ({total_params/1e6:.1f}M)")
        
        # 加载预训练权重
        pretrain_path = 'pretrained_models/mae_pretrain_vit_base.pth'
        if os.path.exists(pretrain_path):
            print("  🔄 加载预训练权重...")
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            print("  ✅ 预训练权重加载成功")
            
            # 移动到设备并测试推理
            model = model.to(device)
            model.eval()
            
            print("  🔄 测试推理...")
            with torch.no_grad():
                x = torch.randn(1, 3, 224, 224, device=device)
                loss, pred, mask = model(x, mask_ratio=0.75)
            
            print(f"  ✅ 推理测试成功:")
            print(f"    输入形状: {x.shape}")
            print(f"    损失值: {loss.item():.4f}")
            print(f"    预测形状: {pred.shape}")
            print(f"    掩码形状: {mask.shape}")
            print(f"    掩码比例: {mask.float().mean().item():.2%}")
            
            return True
        else:
            print(f"  ❌ 预训练模型不存在: {pretrain_path}")
            return False
            
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return False

def show_usage_examples():
    """显示使用示例"""
    print("\n📚 使用示例:")
    print("1. 基本重建演示:")
    print("   python simple_mae_demo.py")
    print()
    print("2. 交互式Jupyter演示:")
    print("   jupyter notebook demo/mae_visualize.ipynb")
    print()
    print("3. 在自己的图片上测试:")
    print("   python visualize_training_results.py")
    print()
    print("4. 微调模型进行分类:")
    print("   python main_finetune.py --model vit_base_patch16 \\")
    print("     --resume pretrained_models/mae_pretrain_vit_base.pth \\")
    print("     --data_path /path/to/your/dataset")

def main():
    """主函数"""
    print("🎭 MAE 预训练模型检查")
    print("=" * 50)
    
    # 检查文件
    files_ok = check_files()
    
    # 检查Python环境
    env_ok = check_python_env()
    
    if not files_ok:
        print("\n❌ 文件检查失败，请先下载模型:")
        print("   ./download_models.sh")
        return
    
    if not env_ok:
        print("\n❌ 环境检查失败，请安装缺少的包")
        return
    
    # 测试模型加载
    model_ok = test_model_loading()
    
    if model_ok:
        print("\n🎉 所有检查通过！MAE预训练模型可以正常使用")
        show_usage_examples()
    else:
        print("\n❌ 模型加载失败")

if __name__ == "__main__":
    main()

