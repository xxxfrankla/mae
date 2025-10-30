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
# 1. 该模块只包含一个核心函数 train_one_epoch，用来执行 MAE 预训练中的单个 epoch。
# 2. 函数会遍历一个数据加载器，前向计算被遮挡的图像，反向传播并更新模型参数。
# 3. 同时负责学习率调度、混合精度缩放、梯度累计、分布式日志统计以及 TensorBoard 记录。
# 4. train_one_epoch 返回一个字典，汇总平均损失和学习率等指标，供主训练脚本保存或打印。
# 5. 该逻辑与原始 MAE 论文在 PyTorch 的实现一致，是整个预训练流程的执行核心。
import math  # 提供数学工具（用于检查损失是否为有限值）
import sys  # 允许在异常情况下直接退出程序
from typing import Iterable  # 类型注解，表示 data_loader 可以是任何可迭代对象

import torch  # PyTorch 主库，处理张量运算与自动微分

import util.misc as misc  # 仓库内通用工具：日志、分布式辅助等
import util.lr_sched as lr_sched  # 自定义学习率调度模块（余弦退火等）


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)  # 将模型切换到训练模式（启用 dropout、BN 的训练行为等）
    metric_logger = misc.MetricLogger(delimiter="  ")  # 创建日志记录器，负责格式化打印训练信息
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 新增一个记录学习率的计量器
    header = 'Epoch: [{}]'.format(epoch)  # 打印前缀，标明当前 epoch 编号
    print_freq = 20  # 每隔多少个 iteration 打印一次日志

    accum_iter = args.accum_iter  # 从外部参数读取梯度累计步数

    optimizer.zero_grad()  # 清空上一轮遗留的梯度，确保本轮从零开始累计

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))  # 如果启用了 TensorBoard，先告知日志目录

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # data_iter_step：当前迭代的编号；samples：一个 batch 的图像；第二个返回值是标签，这里用不到

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:  # 每完成一次梯度更新（考虑累计步数）才调整学习率
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)  # 根据迭代进度计算新的学习率

        samples = samples.to(device, non_blocking=True)  # 将图像张量搬到目标设备（GPU/MPS），non_blocking 可减少等待

        with torch.cuda.amp.autocast():  # 自动混合精度上下文，减少显存占用、提高速度
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)  # 前向传播，MAE 返回重建损失以及一些诊断信息

        loss_value = loss.item()  # 将 loss 张量转换成 Python float，便于日志记录

        if not math.isfinite(loss_value):  # 检查损失是否为非数（NaN）或无穷
            print("Loss is {}, stopping training".format(loss_value))  # 若出现异常损失，打印提示
            sys.exit(1)  # 直接退出，避免继续训练导致模型崩坏

        loss /= accum_iter  # 为了梯度累计，将 loss 平均分配到每次小步更新
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)  # 使用混合精度缩放器管理反传与梯度更新
        if (data_iter_step + 1) % accum_iter == 0:  # 当完成一次完整的累计周期
            optimizer.zero_grad()  # 清空梯度，准备下一轮累计

        torch.cuda.synchronize()  # 等待当前 GPU 任务完成，保证统计值准确（在单 GPU 上也安全）

        metric_logger.update(loss=loss_value)  # 将原始损失值写入日志统计器

        lr = optimizer.param_groups[0]["lr"]  # 从优化器中读取当前学习率（AdamW 只有一个 param_group）
        metric_logger.update(lr=lr)  # 同步学习率到日志

        loss_value_reduce = misc.all_reduce_mean(loss_value)  # 在多进程场景下取所有 GPU 的平均损失
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:  # 仅在真正更新参数的迭代记录 TensorBoard
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)  # 将 epoch 进度放大 1000 倍，使曲线更平滑
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)  # 写入平均损失到 TensorBoard
            log_writer.add_scalar('lr', lr, epoch_1000x)  # 写入学习率到 TensorBoard


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()  # 汇总所有进程的统计信息，确保打印一致
    print("Averaged stats:", metric_logger)  # 打印该 epoch 的平均指标
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}  # 返回一个字典，每个指标取全局平均值
