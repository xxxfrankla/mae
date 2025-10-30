#!/bin/bash
# MAE 预训练模型下载脚本

cd /Users/tdu/Documents/GitHub/mae/pretrained_models

echo "🔽 开始下载 MAE 预训练模型..."

# ViT-Base 预训练模型 (推荐)
echo "📥 下载 ViT-Base 预训练模型..."
curl -L -o mae_pretrain_vit_base.pth https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth

# ViT-Base 微调模型 (用于评估)
echo "📥 下载 ViT-Base 微调模型..."
curl -L -o mae_finetuned_vit_base.pth https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth

echo "✅ 模型下载完成！"
echo "📁 模型保存在: $(pwd)"
ls -lh *.pth
