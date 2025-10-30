#!/bin/bash
# 快速验证配置，5个epoch
# 有效 batch size: 64
# 实际学习率: 3.75e-05

export KMP_DUPLICATE_LIB_OK=TRUE

python main_pretrain.py \
    --batch_size 4 \
    --accum_iter 16 \
    --epochs 5 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --blr 0.00015 \
    --weight_decay 0.05 \
    --warmup_epochs 1 \
    --data_path ./test_dataset \
    --output_dir ./output_m4 \
    --log_dir ./output_m4 \
    --device mps \
    --num_workers 4 \
    --pin_mem