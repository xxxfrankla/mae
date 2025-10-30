#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE

# 尝试修复重建问题
echo '🛠️  尝试修复MAE重建问题...'

# 修复重建问题的尝试
python main_pretrain_animediffusion.py \
    --mask_ratio 0.25 \
    --epochs 50 \
    --batch_size 4 \
    --accum_iter 16 \
    --blr 1e-5 \
    --warmup_epochs 15 \
    --max_samples 1000 \
    --weight_decay 0.01 \
    --output_dir ./output_fix_attempt \
    --log_dir ./output_fix_attempt \
    --norm_pix_loss  # 尝试关闭这个选项
