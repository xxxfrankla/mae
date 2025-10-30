# -*- coding: utf-8 -*-  # 指定文件编码为UTF-8，便于中文注释显示
# Copyright (c) Meta Platforms, Inc. and affiliates.  # Meta平台及其附属公司版权所有
# All rights reserved.  # 保留所有权利

# This source code is licensed under the license found in the  # 此源代码根据根目录中的LICENSE文件授权
# LICENSE file in the root directory of this source tree.  # 具体许可信息请查看根目录的LICENSE文件
# --------------------------------------------------------  # 分隔线，便于阅读
# References:  # 参考资料说明
# DeiT: https://github.com/facebookresearch/deit  # DeiT项目链接
# BEiT: https://github.com/microsoft/unilm/tree/master/beit  # BEiT项目链接
# --------------------------------------------------------  # 分隔线，便于阅读

import argparse  # 导入argparse用于解析命令行参数
import datetime  # 导入datetime用于处理时间和日期
import json  # 导入json用于读写日志文件
import numpy as np  # 导入numpy用于数值运算
import os  # 导入os用于文件和路径操作
import time  # 导入time用于计时
from pathlib import Path  # 从pathlib导入Path用于处理路径对象

import torch  # 导入PyTorch主库
import torch.backends.cudnn as cudnn  # 导入cudnn后端配置接口
from torch.utils.tensorboard import SummaryWriter  # 导入TensorBoard写入器

import timm  # 导入timm模型库

assert timm.__version__ == "0.3.2"  # version check  # 确保使用的timm版本为0.3.2
from timm.models.layers import trunc_normal_  # 从timm导入截断正态初始化函数
from timm.data.mixup import Mixup  # 导入Mixup数据增强工具
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy  # 导入两种交叉熵损失

import util.lr_decay as lrd  # 导入自定义的学习率衰减模块并取别名lrd
import util.misc as misc  # 导入自定义工具集misc
from util.datasets import build_dataset  # 导入数据集构建函数
from util.pos_embed import interpolate_pos_embed  # 导入位置编码插值函数
from util.misc import NativeScalerWithGradNormCount as NativeScaler  # 导入原生混合精度缩放器

import models_vit  # 导入自定义Vision Transformer模型集合

from engine_finetune import train_one_epoch, evaluate  # 导入单轮训练与评估函数


def get_args_parser():  # 定义获取参数解析器的函数
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)  # 创建参数解析器并关闭默认帮助
    parser.add_argument('--batch_size', default=64, type=int,  # 添加批量大小参数
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')  # 设置帮助信息
    parser.add_argument('--epochs', default=50, type=int)  # 添加训练轮数参数
    parser.add_argument('--accum_iter', default=1, type=int,  # 添加梯度累积迭代次数
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')  # 说明梯度累积的作用

    # Model parameters  # 模型参数组
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',  # 添加模型名称参数
                        help='Name of model to train')  # 说明此参数指定训练的模型

    parser.add_argument('--input_size', default=224, type=int,  # 设置输入图像尺寸
                        help='images input size')  # 帮助信息说明输入尺寸

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',  # 设置DropPath比例
                        help='Drop path rate (default: 0.1)')  # 帮助信息说明默认值

    # Optimizer parameters  # 优化器参数组
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',  # 设置梯度裁剪阈值
                        help='Clip gradient norm (default: None, no clipping)')  # 帮助信息说明默认不裁剪
    parser.add_argument('--weight_decay', type=float, default=0.05,  # 设置权重衰减系数
                        help='weight decay (default: 0.05)')  # 帮助信息说明默认值

    parser.add_argument('--lr', type=float, default=None, metavar='LR',  # 设置绝对学习率
                        help='learning rate (absolute lr)')  # 帮助信息说明此参数
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',  # 设置基准学习率
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')  # 说明绝对学习率计算方式
    parser.add_argument('--layer_decay', type=float, default=0.75,  # 设置层级学习率衰减系数
                        help='layer-wise lr decay from ELECTRA/BEiT')  # 说明来源于相关工作

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',  # 设置最小学习率
                        help='lower lr bound for cyclic schedulers that hit 0')  # 说明用于循环调度器的下限

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',  # 设置学习率预热轮数
                        help='epochs to warmup LR')  # 帮助信息说明用途

    # Augmentation parameters  # 数据增强参数组
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',  # 设置颜色抖动幅度
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')  # 说明何时启用
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',  # 设置自动增强策略
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),  # 描述可选策略
    parser.add_argument('--smoothing', type=float, default=0.1,  # 设置标签平滑系数
                        help='Label smoothing (default: 0.1)')  # 说明默认值

    # * Random Erase params  # 随机擦除参数
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',  # 设置随机擦除概率
                        help='Random erase prob (default: 0.25)')  # 说明默认值
    parser.add_argument('--remode', type=str, default='pixel',  # 设置随机擦除模式
                        help='Random erase mode (default: "pixel")')  # 说明默认模式
    parser.add_argument('--recount', type=int, default=1,  # 设置随机擦除次数
                        help='Random erase count (default: 1)')  # 说明默认次数
    parser.add_argument('--resplit', action='store_true', default=False,  # 设置是否在第一个增强前避免擦除
                        help='Do not random erase first (clean) augmentation split')  # 说明作用

    # * Mixup params  # Mixup相关参数
    parser.add_argument('--mixup', type=float, default=0,  # 设置Mixup系数
                        help='mixup alpha, mixup enabled if > 0.')  # 说明大于0才启用
    parser.add_argument('--cutmix', type=float, default=0,  # 设置CutMix系数
                        help='cutmix alpha, cutmix enabled if > 0.')  # 说明大于0才启用
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,  # 设置CutMix的面积范围
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')  # 说明覆盖alpha参数
    parser.add_argument('--mixup_prob', type=float, default=1.0,  # 设置执行Mixup或CutMix的概率
                        help='Probability of performing mixup or cutmix when either/both is enabled')  # 说明概率意义
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,  # 设置在Mixup和CutMix之间切换的概率
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')  # 说明作用
    parser.add_argument('--mixup_mode', type=str, default='batch',  # 设置Mixup模式
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')  # 说明可选模式

    # * Finetuning params  # 微调相关参数
    parser.add_argument('--finetune', default='',  # 设置微调时加载的模型路径
                        help='finetune from checkpoint')  # 说明需要指定检查点
    parser.add_argument('--global_pool', action='store_true')  # 设置是否使用全局池化输出
    parser.set_defaults(global_pool=True)  # 默认开启全局池化
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',  # 当指定时使用CLS Token输出
                        help='Use class token instead of global pool for classification')  # 说明重定向global_pool标志

    # Dataset parameters  # 数据集相关参数
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,  # 数据集根目录
                        help='dataset path')  # 说明需要正确设置路径
    parser.add_argument('--nb_classes', default=1000, type=int,  # 分类类别数
                        help='number of the classification types')  # 说明类别数量

    parser.add_argument('--output_dir', default='./output_dir',  # 输出目录
                        help='path where to save, empty for no saving')  # 说明为空则不保存
    parser.add_argument('--log_dir', default='./output_dir',  # 日志目录
                        help='path where to tensorboard log')  # 说明TensorBoard日志路径
    parser.add_argument('--device', default='cuda',  # 训练设备类型
                        help='device to use for training / testing')  # 说明可为cuda或cpu
    parser.add_argument('--seed', default=0, type=int)  # 随机种子设置
    parser.add_argument('--resume', default='',  # 断点恢复模型路径
                        help='resume from checkpoint')  # 说明用于恢复训练

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',  # 训练起始轮数
                        help='start epoch')  # 说明可用于断点续训
    parser.add_argument('--eval', action='store_true',  # 设置只进行评估
                        help='Perform evaluation only')  # 说明启用后不训练
    parser.add_argument('--dist_eval', action='store_true', default=False,  # 设置是否分布式评估
                        help='Enabling distributed evaluation (recommended during training for faster monitor')  # 说明分布式评估优势
    parser.add_argument('--num_workers', default=10, type=int)  # DataLoader工作线程数
    parser.add_argument('--pin_mem', action='store_true',  # 设置DataLoader是否锁页内存
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')  # 说明性能影响
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')  # 提供关闭pin_mem的开关
    parser.set_defaults(pin_mem=True)  # 默认开启pin_mem

    # distributed training parameters  # 分布式训练参数
    parser.add_argument('--world_size', default=1, type=int,  # 进程总数
                        help='number of distributed processes')  # 说明用于分布式训练
    parser.add_argument('--local_rank', default=-1, type=int)  # 当前进程的本地rank
    parser.add_argument('--dist_on_itp', action='store_true')  # 是否在ImageNet训练平台上运行
    parser.add_argument('--dist_url', default='env://',  # 分布式初始化URL
                        help='url used to set up distributed training')  # 说明常用方式为环境变量

    return parser  # 返回配置好的参数解析器


def main(args):  # 主函数，负责执行训练或评估流程
    misc.init_distributed_mode(args)  # 根据参数初始化分布式环境

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))  # 打印当前脚本所在目录
    print("{}".format(args).replace(', ', ',\n'))  # 打印参数配置并格式化换行

    device = torch.device(args.device)  # 根据参数选择计算设备

    # fix the seed for reproducibility  # 固定随机种子以保证可复现性
    seed = args.seed + misc.get_rank()  # 使用基本种子加上进程rank得到最终种子
    torch.manual_seed(seed)  # 设置PyTorch的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子

    cudnn.benchmark = True  # 启用cudnn的benchmark模式以自动选择最优算法

    dataset_train = build_dataset(is_train=True, args=args)  # 构建训练数据集
    dataset_val = build_dataset(is_train=False, args=args)  # 构建验证数据集

    if True:  # args.distributed:  # 分布式采样逻辑，示例中固定为True
        num_tasks = misc.get_world_size()  # 获取总进程数
        global_rank = misc.get_rank()  # 获取当前进程rank
        sampler_train = torch.utils.data.DistributedSampler(  # 创建分布式训练采样器
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True  # 指定副本数、rank并打乱数据
        )
        print("Sampler_train = %s" % str(sampler_train))  # 打印训练采样器信息
        if args.dist_eval:  # 如果启用了分布式评估
            if len(dataset_val) % num_tasks != 0:  # 检查验证集大小是否能整除进程数
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '  # 打印警告提示
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '  # 说明可能的影响
                      'equal num of samples per-process.')  # 完成警告说明
            sampler_val = torch.utils.data.DistributedSampler(  # 创建分布式验证采样器
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True以减少监控偏差
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)  # 否则使用顺序采样器
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)  # 非分布式时使用随机采样
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)  # 验证集使用顺序采样

    if global_rank == 0 and args.log_dir is not None and not args.eval:  # 仅在主进程且需要训练时管理日志
        os.makedirs(args.log_dir, exist_ok=True)  # 创建日志目录
        log_writer = SummaryWriter(log_dir=args.log_dir)  # 初始化TensorBoard写入器
    else:
        log_writer = None  # 其他情况下不写日志

    data_loader_train = torch.utils.data.DataLoader(  # 创建训练数据加载器
        dataset_train, sampler=sampler_train,  # 设置数据集和采样器
        batch_size=args.batch_size,  # 指定批量大小
        num_workers=args.num_workers,  # 指定工作线程数
        pin_memory=args.pin_mem,  # 是否锁定内存
        drop_last=True,  # 丢弃最后一个不完整的批次
    )

    data_loader_val = torch.utils.data.DataLoader(  # 创建验证数据加载器
        dataset_val, sampler=sampler_val,  # 指定数据集和采样器
        batch_size=args.batch_size,  # 验证批量大小沿用相同设置
        num_workers=args.num_workers,  # 验证时同样使用指定线程数
        pin_memory=args.pin_mem,  # 验证时同样控制锁页内存
        drop_last=False  # 不丢弃最后一个批次
    )

    mixup_fn = None  # 初始化Mixup函数占位符
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None  # 判断是否需要启用Mixup或CutMix
    if mixup_active:  # 如果需要应用Mixup/CutMix
        print("Mixup is activated!")  # 打印提示信息
        mixup_fn = Mixup(  # 创建Mixup实例
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,  # 设置Mixup/CutMix相关参数
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,  # 设置执行概率和模式
            label_smoothing=args.smoothing, num_classes=args.nb_classes)  # 指定标签平滑和类别数
    
    model = models_vit.__dict__[args.model](  # 根据名称从模型字典中实例化模型
        num_classes=args.nb_classes,  # 指定分类头输出类别数
        drop_path_rate=args.drop_path,  # 指定DropPath比例
        global_pool=args.global_pool,  # 指定是否使用全局池化
    )

    if args.finetune and not args.eval:  # 当提供微调检查点且非纯评估模式时
        checkpoint = torch.load(args.finetune, map_location='cpu')  # 从CPU加载预训练检查点

        print("Load pre-trained checkpoint from: %s" % args.finetune)  # 打印加载信息
        checkpoint_model = checkpoint['model']  # 取出模型参数字典
        state_dict = model.state_dict()  # 获取当前模型的参数字典
        for k in ['head.weight', 'head.bias']:  # 遍历分类头的权重和偏置
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:  # 若形状不匹配
                print(f"Removing key {k} from pretrained checkpoint")  # 提示删除对应键
                del checkpoint_model[k]  # 从预训练字典中删除该参数

        # interpolate position embedding  # 处理位置编码插值
        interpolate_pos_embed(model, checkpoint_model)  # 调整位置编码以适配新模型

        # load pre-trained model  # 加载预训练权重
        msg = model.load_state_dict(checkpoint_model, strict=False)  # 以非严格模式载入参数
        print(msg)  # 打印加载结果信息

        if args.global_pool:  # 如果使用全局池化
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}  # 检查缺失键是否符合预期
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}  # 未使用全局池化时的缺失键

        # manually initialize fc layer  # 手动初始化分类头
        trunc_normal_(model.head.weight, std=2e-5)  # 使用截断正态分布初始化权重

    model.to(device)  # 将模型移动到指定设备

    model_without_ddp = model  # 保存未包装的模型引用
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 统计可训练参数数量

    print("Model = %s" % str(model_without_ddp))  # 打印模型结构
    print('number of params (M): %.2f' % (n_parameters / 1.e6))  # 打印参数量（单位百万）

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()  # 计算有效批量大小
    
    if args.lr is None:  # only base_lr is specified  # 如果未设置绝对学习率
        args.lr = args.blr * eff_batch_size / 256  # 用基准学习率推算实际学习率

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))  # 打印基准学习率
    print("actual lr: %.2e" % args.lr)  # 打印实际学习率

    print("accumulate grad iterations: %d" % args.accum_iter)  # 打印梯度累积次数
    print("effective batch size: %d" % eff_batch_size)  # 打印有效批量大小

    if args.distributed:  # 若启用分布式训练
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])  # 使用DDP包装模型
        model_without_ddp = model.module  # 更新未封装模型引用

    # build optimizer with layer-wise lr decay (lrd)  # 使用层级学习率衰减构建优化器
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,  # 按层生成参数组
        no_weight_decay_list=model_without_ddp.no_weight_decay(),  # 指定无需权重衰减的参数
        layer_decay=args.layer_decay  # 设置层级衰减比例
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)  # 使用AdamW优化器
    loss_scaler = NativeScaler()  # 创建混合精度缩放器

    if mixup_fn is not None:  # 若启用了Mixup
        # smoothing is handled with mixup label transform  # 标签平滑由Mixup内部处理
        criterion = SoftTargetCrossEntropy()  # 使用软目标交叉熵
    elif args.smoothing > 0.:  # 若未启用Mixup但设置了标签平滑
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)  # 使用标签平滑交叉熵
    else:
        criterion = torch.nn.CrossEntropyLoss()  # 默认使用标准交叉熵

    print("criterion = %s" % str(criterion))  # 打印损失函数信息

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)  # 尝试加载断点

    if args.eval:  # 若指定只评估
        test_stats = evaluate(data_loader_val, model, device)  # 在验证集上评估模型
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")  # 打印Top-1准确率
        exit(0)  # 评估完成后退出程序

    print(f"Start training for {args.epochs} epochs")  # 打印训练开始提示
    start_time = time.time()  # 记录训练起始时间
    max_accuracy = 0.0  # 初始化最高准确率
    for epoch in range(args.start_epoch, args.epochs):  # 循环遍历每个训练轮次
        if args.distributed:  # 若启用分布式
            data_loader_train.sampler.set_epoch(epoch)  # 为采样器设置当前轮次，以保证shuffle一致性
        train_stats = train_one_epoch(  # 训练一个epoch并获取统计信息
            model, criterion, data_loader_train,  # 传入模型、损失函数和数据加载器
            optimizer, device, epoch, loss_scaler,  # 传入优化器、设备、当前轮次和缩放器
            args.clip_grad, mixup_fn,  # 传入梯度裁剪阈值和Mixup函数
            log_writer=log_writer,  # 传入日志记录器
            args=args  # 传入全部参数
        )
        if args.output_dir:  # 若需要保存模型
            misc.save_model(  # 调用保存函数
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,  # 保存模型和优化器状态
                loss_scaler=loss_scaler, epoch=epoch)  # 同步保存缩放器和当前轮次

        test_stats = evaluate(data_loader_val, model, device)  # 每个轮次后进行验证
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")  # 打印当前准确率
        max_accuracy = max(max_accuracy, test_stats["acc1"])  # 更新最高准确率
        print(f'Max accuracy: {max_accuracy:.2f}%')  # 打印历史最佳准确率

        if log_writer is not None:  # 若启用了日志记录
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)  # 记录Top-1准确率
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)  # 记录Top-5准确率
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)  # 记录验证损失

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},  # 构建包含训练指标的字典
                        **{f'test_{k}': v for k, v in test_stats.items()},  # 合并验证指标
                        'epoch': epoch,  # 记录当前轮次
                        'n_parameters': n_parameters}  # 记录模型参数数量

        if args.output_dir and misc.is_main_process():  # 若主进程负责写日志
            if log_writer is not None:  # 如果TensorBoard写入器存在则刷新
                log_writer.flush()  # 刷新缓冲区
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:  # 追加写入日志文件
                f.write(json.dumps(log_stats) + "\n")  # 将统计信息写入文本

    total_time = time.time() - start_time  # 计算训练总耗时
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))  # 将秒数转换为可读格式
    print('Training time {}'.format(total_time_str))  # 打印训练总时间


if __name__ == '__main__':  # 主程序入口
    args = get_args_parser()  # 构建参数解析器
    args = args.parse_args()  # 解析命令行参数
    if args.output_dir:  # 若指定了输出目录
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)  # 确保目录存在
    main(args)  # 调用主函数开始流程
