#!/usr/bin/env python3
"""
在 AnimeDiffusion 数据集上进行 MAE 预训练
优化处理高分辨率动漫图片
"""

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image

import timm
assert timm.__version__ == "0.3.2"
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc_mps import NativeScalerWithGradNormCount as NativeScaler

import models_mae
from engine_pretrain_mps import train_one_epoch

class AnimeDiffusionDataset(Dataset):
    """AnimeDiffusion 数据集包装器"""
    
    def __init__(self, hf_dataset, transform=None, max_samples=None):
        self.dataset = hf_dataset
        self.transform = transform
        
        if max_samples is not None:
            self.length = min(max_samples, len(hf_dataset))
        else:
            self.length = len(hf_dataset)
        
        print(f"📊 AnimeDiffusion 数据集: {self.length} 张图片 (原始: 1920×1080)")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            sample = self.dataset[idx]
            image = sample['image']
            
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    image = Image.new('RGB', (1920, 1080), color='black')
            
            # 处理RGBA
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, 0
            
        except Exception as e:
            print(f"警告: 加载样本 {idx} 时出错: {e}")
            default_img = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                default_img = self.transform(default_img)
            return default_img, 0

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training on AnimeDiffusion Dataset', add_help=False)
    
    # 训练参数
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--accum_iter', default=8, type=int,
                        help='Accumulate gradient iterations')
    parser.add_argument('--epochs', default=20, type=int)
    
    # 模型参数
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str,
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size (fixed at 224 for MAE)')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use normalized pixels as targets')
    parser.set_defaults(norm_pix_loss=True)
    
    # 优化器参数
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=1.5e-4,
                        help='base learning rate')
    parser.add_argument('--min_lr', type=float, default=0.)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    # 数据集参数
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to use (None for all 8202)')
    parser.add_argument('--resize_strategy', default='smart_crop', 
                        choices=['smart_crop', 'center_crop', 'resize_only'],
                        help='How to handle 1920x1080 -> 224x224')
    
    # 输出参数
    parser.add_argument('--output_dir', default='./output_animediffusion')
    parser.add_argument('--log_dir', default='./output_animediffusion')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--save_freq', default=5, type=int)
    
    return parser

def create_animediffusion_dataloader(args):
    """创建 AnimeDiffusion 数据加载器"""
    
    print(f"🎌 创建 AnimeDiffusion 数据加载器...")
    print(f"  原始分辨率: 1920×1080")
    print(f"  目标分辨率: {args.input_size}×{args.input_size}")
    print(f"  缩放策略: {args.resize_strategy}")
    print(f"  批次大小: {args.batch_size}")
    
    # 加载数据集
    try:
        ds = load_dataset("Mercity/AnimeDiffusion_Dataset")
        train_dataset = ds['train']
        print(f"✅ 数据集加载成功: {len(train_dataset)} 张图片")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return None, None
    
    # 根据策略定义变换
    if args.resize_strategy == 'smart_crop':
        # 智能裁剪：先缩放到稍大尺寸，再随机裁剪
        transform = transforms.Compose([
            transforms.Resize(int(args.input_size * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(args.input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif args.resize_strategy == 'center_crop':
        # 中心裁剪
        transform = transforms.Compose([
            transforms.Resize(int(args.input_size * 1.1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # resize_only
        # 直接缩放
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # 创建数据集
    anime_dataset = AnimeDiffusionDataset(train_dataset, transform=transform, max_samples=args.max_samples)
    
    # 创建数据加载器
    dataloader = DataLoader(
        anime_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True
    )
    
    print(f"✅ 数据加载器创建成功: {len(dataloader)} 个批次")
    
    return dataloader, anime_dataset

def main(args):
    print('🎌 MAE AnimeDiffusion 数据集预训练')
    print('=' * 60)
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    print(f'🔧 使用设备: {device}')

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # 创建数据加载器
    data_loader_train, dataset_train = create_animediffusion_dataloader(args)
    
    if data_loader_train is None:
        print("❌ 数据加载器创建失败")
        return
    
    # 创建模型
    print(f"\n🤖 创建 {args.model} 模型...")
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)
    
    # 计算学习率
    eff_batch_size = args.batch_size * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"\n📈 训练配置:")
    print(f"  数据集: AnimeDiffusion ({len(dataset_train)} 张图片)")
    print(f"  分辨率: 1920×1080 → {args.input_size}×{args.input_size}")
    print(f"  缩放策略: {args.resize_strategy}")
    print(f"  有效批次大小: {eff_batch_size}")
    print(f"  学习率: {args.lr:.2e}")
    print(f"  掩码比例: {args.mask_ratio}")

    # 创建优化器
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    
    # 创建损失缩放器
    loss_scaler = NativeScaler(device_type=device.type)

    # 创建日志记录器
    log_writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir else None

    print(f"\n🚀 开始训练 {args.epochs} 个epoch...")
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(args.start_epoch, args.epochs):
        print(f"\n📅 Epoch {epoch+1}/{args.epochs}")
        
        # 训练一个epoch
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        
        # 更新最佳损失
        current_loss = train_stats['loss']
        if current_loss < best_loss:
            best_loss = current_loss
            print(f"🎉 新的最佳损失: {best_loss:.4f}")
        
        # 保存检查点
        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            print(f"💾 保存检查点: epoch {epoch}")
            misc.save_model(
                args=args, model=model, model_without_ddp=model, 
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        # 记录日志
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
            'best_loss': best_loss
        }

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # 显示进度
        if epoch > 0:
            elapsed_time = time.time() - start_time
            remaining_epochs = args.epochs - epoch - 1
            eta = elapsed_time / (epoch + 1 - args.start_epoch) * remaining_epochs
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            print(f"⏱️  预计剩余时间: {eta_str}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print(f'\n🎉 AnimeDiffusion 预训练完成!')
    print(f'⏱️  总训练时间: {total_time_str}')
    print(f'🏆 最佳损失: {best_loss:.4f}')
    print(f'📁 模型保存在: {args.output_dir}')
    
    # 生成训练总结
    summary = {
        'dataset': 'AnimeDiffusion',
        'dataset_size': len(dataset_train),
        'original_resolution': '1920x1080',
        'target_resolution': f'{args.input_size}x{args.input_size}',
        'resize_strategy': args.resize_strategy,
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'effective_batch_size': eff_batch_size,
        'learning_rate': args.lr,
        'mask_ratio': args.mask_ratio,
        'best_loss': best_loss,
        'training_time': total_time_str,
        'device': str(device)
    }
    
    summary_path = os.path.join(args.output_dir, 'animediffusion_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 训练总结保存: {summary_path}")

if __name__ == '__main__':
    # 设置环境变量
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    args = get_args_parser()
    args = args.parse_args()
    
    # 强制使用224×224（MAE模型限制）
    if args.input_size != 224:
        print(f"⚠️  MAE模型只支持224×224，自动调整 {args.input_size} → 224")
        args.input_size = 224
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)


