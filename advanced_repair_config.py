#!/usr/bin/env python3
"""
高级图像修复配置生成器
针对模糊问题的深度优化
"""

def create_advanced_repair_configs():
    """创建高级图像修复配置"""
    
    configs = {
        # 超长训练配置
        'ultra_long_training': {
            'mask_ratio': 0.25,
            'epochs': 200,
            'batch_size': 4,
            'accum_iter': 16,
            'blr': 3e-5,  # 更低的学习率
            'warmup_epochs': 30,
            'min_lr': 5e-7,
            'max_samples': 3000,
            'weight_decay': 0.01,
            'description': '超长训练 - 追求最佳重建质量'
        },
        
        # 渐进式掩码训练
        'progressive_mask': {
            'mask_ratio': 0.15,  # 从更简单的任务开始
            'epochs': 50,
            'batch_size': 6,
            'accum_iter': 12,
            'blr': 4e-5,
            'warmup_epochs': 15,
            'max_samples': 2000,
            'description': '渐进式掩码 - 从15%开始'
        },
        
        # 大批次稳定训练
        'stable_large_batch': {
            'mask_ratio': 0.25,
            'epochs': 80,
            'batch_size': 2,  # 更小批次
            'accum_iter': 32, # 更大累积
            'blr': 2e-5,      # 更低学习率
            'warmup_epochs': 25,
            'max_samples': 4000,
            'weight_decay': 0.005,
            'description': '稳定大批次训练'
        }
    }
    
    print("🎯 高级图像修复配置:")
    print("=" * 80)
    
    for name, config in configs.items():
        print(f"\n📋 {name}:")
        print(f"  描述: {config['description']}")
        print(f"  掩码比例: {config['mask_ratio']*100:.0f}%")
        print(f"  训练轮数: {config['epochs']}")
        print(f"  有效批次: {config['batch_size'] * config['accum_iter']}")
        print(f"  学习率: {config['blr']:.1e}")
        
        # 估算训练时间
        estimated_time = config['epochs'] * 3.5  # 基于之前的观察，每epoch约3.5分钟
        hours = estimated_time // 60
        minutes = estimated_time % 60
        print(f"  预计时间: {hours:.0f}小时{minutes:.0f}分钟")
        
        # 生成命令
        cmd = f"python main_pretrain_animediffusion.py \\\n"
        cmd += f"    --mask_ratio {config['mask_ratio']} \\\n"
        cmd += f"    --epochs {config['epochs']} \\\n"
        cmd += f"    --batch_size {config['batch_size']} \\\n"
        cmd += f"    --accum_iter {config['accum_iter']} \\\n"
        cmd += f"    --blr {config['blr']:.1e} \\\n"
        cmd += f"    --warmup_epochs {config['warmup_epochs']} \\\n"
        cmd += f"    --max_samples {config['max_samples']} \\\n"
        cmd += f"    --weight_decay {config.get('weight_decay', 0.02)} \\\n"
        cmd += f"    --min_lr {config.get('min_lr', 1e-6):.1e} \\\n"
        cmd += f"    --output_dir ./output_{name} \\\n"
        cmd += f"    --log_dir ./output_{name}"
        
        print(f"  命令: {cmd}")
    
    return configs

def create_immediate_test_config():
    """创建立即可测试的快速改进配置"""
    
    print(f"\n🚀 立即可测试的快速改进配置:")
    
    quick_config = {
        'mask_ratio': 0.2,   # 稍微降低难度
        'epochs': 30,        # 适中的训练时间
        'batch_size': 4,
        'accum_iter': 16,
        'blr': 3e-5,         # 更保守的学习率
        'warmup_epochs': 10,
        'max_samples': 1500,
        'weight_decay': 0.01,
        'min_lr': 5e-7
    }
    
    cmd = f"""# 快速改进测试 (约1.5小时)
python main_pretrain_animediffusion.py \\
    --mask_ratio {quick_config['mask_ratio']} \\
    --epochs {quick_config['epochs']} \\
    --batch_size {quick_config['batch_size']} \\
    --accum_iter {quick_config['accum_iter']} \\
    --blr {quick_config['blr']:.1e} \\
    --warmup_epochs {quick_config['warmup_epochs']} \\
    --max_samples {quick_config['max_samples']} \\
    --weight_decay {quick_config['weight_decay']} \\
    --min_lr {quick_config['min_lr']:.1e} \\
    --output_dir ./output_quick_repair \\
    --log_dir ./output_quick_repair"""
    
    print(cmd)
    
    # 保存快速配置
    with open('quick_repair_config.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("export KMP_DUPLICATE_LIB_OK=TRUE\n\n")
        f.write(cmd)
    
    os.chmod('quick_repair_config.sh', 0o755)
    print(f"\n✅ 快速配置保存: quick_repair_config.sh")

def main():
    """主函数"""
    print("🎯 高级图像修复配置生成器")
    print("=" * 50)
    
    # 1. 创建高级配置
    advanced_configs = create_advanced_repair_configs()
    
    # 2. 创建快速测试配置
    create_immediate_test_config()
    
    print(f"\n💡 解决模糊问题的策略:")
    print(f"  🔴 核心问题: 训练时间不够充分")
    print(f"  🟡 次要问题: 学习率和参数调优")
    print(f"  🟢 MAE限制: 本身不是专门的修复模型")
    
    print(f"\n🎯 推荐行动:")
    print(f"  1️⃣ 立即尝试: bash quick_repair_config.sh (1.5小时)")
    print(f"  2️⃣ 如果有时间: 尝试 ultra_long_training 配置")
    print(f"  3️⃣ 考虑替代: 专门的图像修复模型 (如 DDPM, Stable Diffusion inpainting)")

if __name__ == "__main__":
    main()


