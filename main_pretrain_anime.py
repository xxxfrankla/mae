#!/usr/bin/env python3
"""
在动漫数据集上进行 MAE 预训练
专为 Apple M4 和动漫数据优化
"""

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
assert timm.__version__ == "0.3.2"
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc_mps import NativeScalerWithGradNormCount as NativeScaler

import models_mae
from engine_pretrain_mps import train_one_epoch
from anime_dataset_loader import create_anime_dataloader

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training on Anime Dataset', add_help=False)
    
    # 训练参数
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--accum_iter', default=16, type=int,
                        help='Accumulate gradient iterations')
    parser.add_argument('--epochs', default=50, type=int)
    
    # 模型参数
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str,
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets')
    parser.set_defaults(norm_pix_loss=True)  # 默认使用归一化像素损失
    
    # 优化器参数
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    
    # 数据集参数
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use (for testing)')
    parser.add_argument('--output_dir', default='./output_anime',
                        help='path where to save')
    parser.add_argument('--log_dir', default='./output_anime',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='mps',
                        help='device to use for training')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    
    # 保存参数
    parser.add_argument('--save_freq', default=10, type=int,
                        help='Save checkpoint every N epochs')
    
    return parser

def main(args):
    print('🎌 MAE 动漫数据集预训练开始')
    print('=' * 60)
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    print(f'🔧 使用设备: {device}')

    # 设置随机种子
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # 创建动漫数据加载器
    print(f"\n📥 创建动漫数据加载器...")
    data_loader_train, dataset_train = create_anime_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        input_size=args.input_size
    )
    
    if data_loader_train is None:
        print("❌ 数据加载器创建失败")
        return
    
    print(f'📊 数据集统计:')
    print(f'  总样本数: {len(dataset_train)}')
    print(f'  批次数: {len(data_loader_train)}')
    print(f'  每批次样本: {args.batch_size}')
    
    # 创建模型
    print(f"\n🤖 创建 {args.model} 模型...")
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)
    
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    
    # 计算有效批次大小和学习率
    eff_batch_size = args.batch_size * args.accum_iter
    
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"\n📈 训练参数:")
    print(f"  基础学习率: {args.blr:.2e}")
    print(f"  实际学习率: {args.lr:.2e}")
    print(f"  梯度累积步数: {args.accum_iter}")
    print(f"  有效批次大小: {eff_batch_size}")
    print(f"  掩码比例: {args.mask_ratio}")

    # 创建优化器
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(f"\n🔧 优化器: {optimizer}")
    
    # 创建损失缩放器
    loss_scaler = NativeScaler(device_type=device.type)

    # 加载检查点
    misc.load_model(args=args, model_without_ddp=model_without_ddp, 
                   optimizer=optimizer, loss_scaler=loss_scaler)

    # 创建日志记录器
    if args.log_dir is not None:
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

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
        
        # 记录最佳损失
        current_loss = train_stats['loss']
        if current_loss < best_loss:
            best_loss = current_loss
            print(f"🎉 新的最佳损失: {best_loss:.4f}")
        
        # 保存检查点
        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            print(f"💾 保存检查点: epoch {epoch}")
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, 
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        # 记录日志
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
            'best_loss': best_loss
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # 显示进度
        elapsed_time = time.time() - start_time
        remaining_epochs = args.epochs - epoch - 1
        if epoch > 0:
            eta = elapsed_time / (epoch + 1 - args.start_epoch) * remaining_epochs
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            print(f"⏱️  预计剩余时间: {eta_str}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print(f'\n🎉 训练完成!')
    print(f'⏱️  总训练时间: {total_time_str}')
    print(f'🏆 最佳损失: {best_loss:.4f}')
    print(f'📁 模型保存在: {args.output_dir}')
    
    # 生成训练总结
    summary = {
        'dataset': 'anime-captions',
        'total_samples': len(dataset_train) if 'dataset_train' in locals() else 'unknown',
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
    
    summary_path = os.path.join(args.output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 训练总结保存: {summary_path}")

if __name__ == '__main__':
    # 设置环境变量
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    args = get_args_parser()
    args = args.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
