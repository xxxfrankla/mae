# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
# 文件概览说明：
# 1. 通过 argparse 定义命令行参数，使得训练配置可以在启动脚本时灵活修改。
# 2. 初始化分布式训练环境、随机数种子以及数据增强流水线，准备好 ImageNet 数据集。
# 3. 构建 MAE 模型、优化器和混合精度工具，按需从断点恢复训练状态。
# 4. 在指定轮数内循环，调用 engine_pretrain.train_one_epoch 完成单轮训练并记录指标。
# 5. 定期保存模型权重与训练日志，最终打印总耗时。整个流程对应论文中的 MAE 预训练阶段。
import argparse  # 标准库；用于解析命令行参数
import datetime  # 标准库；用于格式化输出耗时
import json  # 标准库；用于写出训练日志
import numpy as np  # 第三方库；用于数值运算与生成随机数
import os  # 标准库；处理路径、环境变量
import time  # 标准库；测量训练耗时
from pathlib import Path  # 标准库；便捷地处理文件路径

import torch  # PyTorch 主库；神经网络训练基础
import torch.backends.cudnn as cudnn  # 控制 cuDNN 行为以提升性能
from torch.utils.tensorboard import SummaryWriter  # 写入 TensorBoard 日志
import torchvision.transforms as transforms  # 常用图像数据增强
import torchvision.datasets as datasets  # 提供 ImageNet 等数据集封装

import timm  # 第三方库；复用了 Vision Transformer 的实现

assert timm.__version__ == "0.3.2"  # 检查 timm 版本确保与官方实现一致
import timm.optim.optim_factory as optim_factory  # timm 内的优化器辅助函数

import util.misc as misc  # 仓库内的工具函数集合
from util.misc import NativeScalerWithGradNormCount as NativeScaler  # 自定义混合精度梯度缩放器

import models_mae  # 仓库内定义的 MAE 模型结构

from engine_pretrain import train_one_epoch  # 单轮训练逻辑（前向、反向、日志）


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)  # 创建命令行参数解析器

    # ==== 基础训练超参 ====
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')  # 每张 GPU 的 mini-batch 大小
    parser.add_argument('--epochs', default=400, type=int)  # 预训练总轮数，默认按论文设置跑 400 轮
    parser.add_argument('--accum_iter', default=1, type=int,  # 梯度累计步数，显存不够时可>1
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',  # 指定使用的 MAE 结构（可以换成 base、huge 等变体）
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,  # 输入图像边长，ViT 默认以 224×224 图像训练
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,  # MAE 掩码比例，0.75 表示随机丢 75% 的 patch
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)  # 默认使用原始像素作为重建目标

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,  # L2 权重衰减，控制模型复杂度
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',  # 可直接指定绝对学习率（一般留空）
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',  # 基础学习率，会根据总批量进行线性缩放
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',  # 余弦退火过程中允许的最小学习率
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',  # 预热轮数，前若干轮线性增大学习率
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,  # ImageNet 根目录（需替换成自己的路径）
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',  # 训练权重与日志保存路径（为空则不保存）
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',  # TensorBoard 日志目录，默认与 output_dir 共用
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',  # 计算设备，可设置为 'cpu'、'cuda' 或具体 GPU:0
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)  # 随机子种子
    parser.add_argument('--resume', default='',  # 指定 checkpoint 路径以恢复训练（留空代表从头开始）
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',  # 恢复训练时的起始轮数（resume 时需要同步设置）
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)  # DataLoader workers 数
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')  # 提供关闭 pin_memory 的开关
    parser.set_defaults(pin_mem=True)  # 默认启用 pin_memory（对 GPU 训练通常更快）

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,  # 总进程数
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)  # 当前进程编号
    parser.add_argument('--dist_on_itp', action='store_true')  # Meta 内部 ITP 平台开关，普通环境可忽略
    parser.add_argument('--dist_url', default='env://',  # 分布式初始化地址，默认读取环境变量
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)  # 根据传入参数初始化分布式环境（单机或多机）

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))  # 打印当前脚本所在目录，方便确认运行位置
    print("{}".format(args).replace(', ', ',\n'))  # 打印全部参数设置，便于日志检查

    device = torch.device(args.device)  # 构造训练设备对象（CPU 或 GPU）

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()  # 为不同进程分配唯一随机种子，避免数据重复
    torch.manual_seed(seed)  # 固定 PyTorch 随机性
    np.random.seed(seed)  # 固定 NumPy 随机性

    cudnn.benchmark = True  # 让 cuDNN 自动选择最优卷积实现，提升训练速度

    # simple augmentation
    # Compose 将多个图像变换串联在一起，按顺序依次执行
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 随机裁剪并缩放到输入尺寸；插值 3 表示双三次
            transforms.RandomHorizontalFlip(),  # 50% 概率水平翻转，增强多样性
            transforms.ToTensor(),  # 将 PIL 图像转为 0-1 浮点张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # 按 ImageNet 统计量标准化
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)  # 读取 ImageFolder 格式的 ImageNet 训练集（目录需按类别划分子文件夹）
    print(dataset_train)  # 打印数据集对象信息（类别数、样本量等）

    # 官方脚本总是假设使用分布式（哪怕 world_size=1），因此这里写死为 True
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()  # 总并行任务数（util.misc 内封装了 torch.distributed 接口）
        global_rank = misc.get_rank()  # 当前进程的全局编号
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )  # 分布式采样器：让不同进程遍历数据集的不同切片
        print("Sampler_train = %s" % str(sampler_train))  # 查看采样器的具体配置
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)  # 单 GPU 训练时使用普通随机采样

    if global_rank == 0 and args.log_dir is not None:  # 仅主进程负责创建日志目录与写入 TensorBoard
        os.makedirs(args.log_dir, exist_ok=True)  # 只在主进程创建日志目录
        log_writer = SummaryWriter(log_dir=args.log_dir)  # 创建 TensorBoard 记录器
    else:
        log_writer = None  # 非主进程或未指定日志目录时不写 TensorBoard

    # DataLoader 关键参数说明：
    # sampler：控制取样顺序（此处为分布式采样器）
    # batch_size：单 GPU 每次喂入模型的样本数量
    # num_workers：后台读取数据的进程数
    # pin_memory：是否将张量固定在内存中，加快拷贝到 GPU 的速度
    # drop_last：若最后一个 batch 不够大，是否丢弃（True 可保证每 batch 大小一致）
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )  # 构建分布式 DataLoader
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)  # __dict__ 允许根据字符串动态取出模型工厂函数

    model.to(device)  # 将模型移动到目标设备

    model_without_ddp = model  # 先记录未包裹 DDP 的模型句柄
    print("Model = %s" % str(model_without_ddp))  # 输出模型结构，确认参数量与层数

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()  # 实际总批量
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256  # 按批量线性缩放学习率

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))  # 打印未缩放前的基础学习率，便于与论文对照
    print("actual lr: %.2e" % args.lr)  # 打印当前实际生效的学习率

    print("accumulate grad iterations: %d" % args.accum_iter)  # 告知梯度累计次数
    print("effective batch size: %d" % eff_batch_size)  # 告知综合批量大小（便于比较实验）

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)  # 包裹 DDP 以同步梯度；device_ids 指定当前进程使用的 GPU
        model_without_ddp = model.module  # 取出原始模型供保存/优化器使用
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)  # 按层设置权重衰减（忽略偏置、归一化层）
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))  # 构建 AdamW 优化器
    print(optimizer)  # 打印优化器配置（学习率、动量等）
    loss_scaler = NativeScaler()  # 混合精度梯度缩放器，自动放大/缩小梯度避免下溢

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)  # 若 args.resume 非空，会恢复模型、优化器与缩放器状态

    print(f"Start training for {args.epochs} epochs")  # 提示训练将运行多少轮
    start_time = time.time()  # 记录起始时间，后续统计总耗时
    for epoch in range(args.start_epoch, args.epochs):  # 主训练循环（包含 start_epoch，结束于 epochs-1）
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)  # 设置 epoch，保证各进程采样不同数据
        train_stats = train_one_epoch(  # 调用封装好的单轮训练逻辑，返回损失、学习率等统计
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):  # 定期或最后一次保存模型（每 20 轮保存一次）
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)  # 保存模型参数、优化器状态以及混合精度缩放因子

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}  # 给统计量加 train_ 前缀并附上 epoch，便于后续解析

        if args.output_dir and misc.is_main_process():  # 仅主进程写日志文件，避免多进程同时追加
            if log_writer is not None:
                log_writer.flush()  # 确保 TensorBoard 缓冲区落盘
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:  # 追加写入 JSON 日志
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time  # 统计总训练时长
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))  # 将秒数转为 人类易读的 时:分:秒
    print('Training time {}'.format(total_time_str))  # 打印训练总耗时


if __name__ == '__main__':
    args = get_args_parser()  # 构建包含所有参数定义的解析器
    args = args.parse_args()  # 从命令行读取参数
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)  # parents=True 允许一次性创建多级目录；exist_ok=True 防重建
    main(args)  # 启动训练入口
