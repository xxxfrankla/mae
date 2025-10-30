#!/usr/bin/env python3
"""
Apple M4 24GB MAE 预训练配置生成器
根据硬件限制生成合适的训练参数
"""

import os
import argparse

def create_m4_configs():
    """为不同场景创建 M4 适配的配置"""
    
    configs = {
        # 快速测试配置 (几分钟验证)
        "quick_test": {
            "batch_size": 4,
            "accum_iter": 16,  # 模拟 64 的 batch size
            "epochs": 5,
            "model": "mae_vit_base_patch16",
            "blr": 1.5e-4,
            "warmup_epochs": 1,
            "description": "快速验证配置，5个epoch"
        },
        
        # 小规模实验配置 (几小时)
        "small_experiment": {
            "batch_size": 8,
            "accum_iter": 32,  # 模拟 256 的 batch size
            "epochs": 50,
            "model": "mae_vit_base_patch16", 
            "blr": 1.5e-4,
            "warmup_epochs": 5,
            "description": "小规模实验，50个epoch"
        },
        
        # 中等规模训练 (1-2天)
        "medium_training": {
            "batch_size": 8,
            "accum_iter": 64,  # 模拟 512 的 batch size
            "epochs": 200,
            "model": "mae_vit_base_patch16",
            "blr": 1.5e-4,
            "warmup_epochs": 20,
            "description": "中等规模训练，200个epoch"
        },
        
        # 长期训练配置 (几天到一周)
        "long_training": {
            "batch_size": 6,
            "accum_iter": 85,  # 模拟 ~512 的 batch size
            "epochs": 400,
            "model": "mae_vit_base_patch16",
            "blr": 1.5e-4,
            "warmup_epochs": 40,
            "description": "长期训练，400个epoch"
        }
    }
    
    return configs

def generate_command(config_name, data_path, output_dir="./output_m4"):
    """生成训练命令"""
    configs = create_m4_configs()
    
    if config_name not in configs:
        print(f"❌ 配置 '{config_name}' 不存在")
        print(f"可用配置: {list(configs.keys())}")
        return None
    
    config = configs[config_name]
    
    # 计算有效 batch size
    effective_batch_size = config["batch_size"] * config["accum_iter"]
    actual_lr = config["blr"] * effective_batch_size / 256
    
    command = f"""# {config['description']}
# 有效 batch size: {effective_batch_size}
# 实际学习率: {actual_lr:.2e}

export KMP_DUPLICATE_LIB_OK=TRUE

python main_pretrain.py \\
    --batch_size {config['batch_size']} \\
    --accum_iter {config['accum_iter']} \\
    --epochs {config['epochs']} \\
    --model {config['model']} \\
    --norm_pix_loss \\
    --mask_ratio 0.75 \\
    --blr {config['blr']} \\
    --weight_decay 0.05 \\
    --warmup_epochs {config['warmup_epochs']} \\
    --data_path {data_path} \\
    --output_dir {output_dir} \\
    --log_dir {output_dir} \\
    --device mps \\
    --num_workers 4 \\
    --pin_mem"""
    
    return command, config

def main():
    parser = argparse.ArgumentParser(description='生成 Apple M4 MAE 预训练配置')
    parser.add_argument('--config', choices=['quick_test', 'small_experiment', 'medium_training', 'long_training'],
                       default='quick_test', help='选择配置类型')
    parser.add_argument('--data_path', required=True, help='数据集路径')
    parser.add_argument('--output_dir', default='./output_m4', help='输出目录')
    parser.add_argument('--save_script', action='store_true', help='保存为脚本文件')
    
    args = parser.parse_args()
    
    command, config = generate_command(args.config, args.data_path, args.output_dir)
    
    if command:
        print(f"🍎 Apple M4 MAE 预训练配置: {args.config}")
        print("=" * 60)
        print(command)
        
        if args.save_script:
            script_name = f"run_pretrain_{args.config}.sh"
            with open(script_name, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(command)
            os.chmod(script_name, 0o755)
            print(f"\n✅ 脚本已保存: {script_name}")
        
        # 显示配置说明
        print(f"\n📊 配置详情:")
        print(f"  模型: {config['model']}")
        print(f"  批次大小: {config['batch_size']}")
        print(f"  梯度累积: {config['accum_iter']}")
        print(f"  有效批次: {config['batch_size'] * config['accum_iter']}")
        print(f"  训练轮数: {config['epochs']}")
        print(f"  预热轮数: {config['warmup_epochs']}")
        
        # 估算训练时间
        estimate_time(config)

def estimate_time(config):
    """估算训练时间"""
    print(f"\n⏱️  训练时间估算 (基于 ImageNet-1K):")
    
    # 基于测试结果的估算
    # ViT-Base, batch_size=8, ~100ms/batch on M4
    images_per_epoch = 1281167  # ImageNet train set
    batches_per_epoch = images_per_epoch // (config['batch_size'] * config['accum_iter'])
    time_per_batch = 0.1  # 100ms estimated
    
    time_per_epoch = batches_per_epoch * time_per_batch / 60  # minutes
    total_time = time_per_epoch * config['epochs'] / 60  # hours
    
    print(f"  每轮约: {time_per_epoch:.1f} 分钟")
    print(f"  总计约: {total_time:.1f} 小时 ({total_time/24:.1f} 天)")
    
    if total_time > 48:
        print("  ⚠️  训练时间较长，建议使用 screen 或 tmux")

if __name__ == "__main__":
    main()
