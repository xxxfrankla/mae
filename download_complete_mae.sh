#!/bin/bash
# 下载包含解码器的完整MAE模型

echo "🔽 下载完整MAE模型..."

# 创建模型目录
mkdir -p complete_mae_models
cd complete_mae_models

echo "📥 下载可视化模型（可能包含解码器）..."

# 下载可视化模型
curl -L -o mae_visualize_vit_large.pth https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth

curl -L -o mae_visualize_vit_large_ganloss.pth https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth

echo "✅ 下载完成！"
echo "📁 模型保存在: $(pwd)"
ls -lh *.pth

echo ""
echo "💡 使用方法:"
echo "1. 检查模型内容: python ../check_model_contents.py"
echo "2. 如果包含解码器，可以直接用于重建"
echo "3. 如果不包含，考虑使用HuggingFace的transformers库"
