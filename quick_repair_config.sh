#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE

# 快速改进测试 (约1.5小时)
python main_pretrain_animediffusion.py \
    --mask_ratio 0.2 \
    --epochs 30 \
    --batch_size 4 \
    --accum_iter 16 \
    --blr 3.0e-05 \
    --warmup_epochs 10 \
    --max_samples 1500 \
    --weight_decay 0.01 \
    --min_lr 5.0e-07 \
    --output_dir ./output_quick_repair \
    --log_dir ./output_quick_repair