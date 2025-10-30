#!/bin/bash
# 小规模实验，50个epoch
# 有效 batch size: 256
# 实际学习率: 1.50e-04

export KMP_DUPLICATE_LIB_OK=TRUE

python main_pretrain.py \
    --batch_size 8 \
    --accum_iter 32 \
    --epochs 50 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --blr 0.00015 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --data_path ./test_dataset \
    --output_dir ./output_m4 \
    --log_dir ./output_m4 \
    --device mps \
    --num_workers 4 \
    --pin_mem