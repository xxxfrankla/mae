#!/bin/bash
# MAE 环境设置脚本
# 解决 macOS 上的 OpenMP 冲突问题

export KMP_DUPLICATE_LIB_OK=TRUE
echo "✅ 环境变量已设置: KMP_DUPLICATE_LIB_OK=TRUE"
echo "✅ 现在可以安全运行 MAE 相关脚本了"
