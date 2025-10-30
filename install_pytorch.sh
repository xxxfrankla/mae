#!/bin/bash
# 安装PyTorch和相关依赖的脚本

echo "🔧 安装PyTorch和相关依赖..."

# 检查是否有conda环境
if command -v conda &> /dev/null; then
    echo "📦 检测到conda，使用conda安装..."
    conda install pytorch torchvision -c pytorch -y
elif command -v pip &> /dev/null; then
    echo "📦 使用pip安装..."
    # 为Apple Silicon Mac安装优化版本
    if [[ $(uname -m) == "arm64" ]]; then
        echo "🍎 检测到Apple Silicon，安装MPS支持版本..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    else
        echo "💻 安装标准CPU版本..."
        pip install torch torchvision
    fi
else
    echo "❌ 未找到conda或pip，请手动安装PyTorch"
    exit 1
fi

echo "✅ 安装完成！"
echo "🧪 测试安装..."

python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'MPS可用: {torch.backends.mps.is_available()}')
print('✅ PyTorch安装成功！')
"

