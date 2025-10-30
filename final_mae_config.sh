#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE

echo '🎯 最终MAE图像修复尝试...'
# 最终推荐配置 - 原始像素 + 长时间训练
python main_pretrain_animediffusion.py \
    --mask_ratio 0.2 \
    --epochs 50 \
    --batch_size 4 \
    --accum_iter 16 \
    --blr 2e-5 \
    --warmup_epochs 15 \
    --max_samples 2000 \
    --weight_decay 0.01 \
    --output_dir ./output_final_attempt \
    --log_dir ./output_final_attempt
    # 注意：不加 --norm_pix_loss 标志，默认为False