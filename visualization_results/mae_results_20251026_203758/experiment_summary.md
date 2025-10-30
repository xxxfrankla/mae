# MAE 实验结果报告

**生成时间**: 2025-10-26 20:38:04

**设备**: mps

## 训练统计

- 训练轮数: 2
- 初始损失: 1.6905
- 最终损失: 1.3080
- 损失下降: 22.6%
- 最高学习率: 1.06e-05

## 重建性能

| 掩码比例 | 重建损失 | 平均误差 | 最大误差 |
|---------|---------|---------|----------|
| 25% | 1.8347 | 0.3787 | 1.0000 |
| 50% | 1.8217 | 0.3790 | 1.0000 |
| 75% | 1.8045 | 0.3792 | 1.0000 |
| 90% | 1.7904 | 0.3791 | 1.0000 |

## 生成的文件

### 训练曲线
- `training_curves/training_curves.png`: 损失和学习率曲线

### 数据集样本
- `dataset_samples/dataset_samples_grid.png`: 数据集样本网格
- `dataset_samples/*_sample.png`: 各类别样本

### 重建结果
- `reconstructions/mae_reconstruction_comparison.png`: 不同掩码比例的重建对比

### 分析数据
- `analysis/training_stats.json`: 训练统计数据
- `analysis/reconstruction_stats.json`: 重建统计数据

