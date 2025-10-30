#!/bin/bash
# 图像修复专用MAE训练配置
export KMP_DUPLICATE_LIB_OK=TRUE

# 图像修复专用MAE训练
python main_pretrain_animediffusion.py \
    --mask_ratio 0.25 \
    --epochs 100 \
    --batch_size 4 \
    --accum_iter 16 \
    --blr 5e-05 \
    --warmup_epochs 20 \
    --min_lr 1e-06 \
    --input_size 224 \
    --resize_strategy smart_crop \
    --max_samples 2000 \
    --weight_decay 0.02 \
    --norm_pix_loss True \
    --save_freq 10 \
    --output_dir ./output_image_repair \
    --log_dir ./output_image_repair