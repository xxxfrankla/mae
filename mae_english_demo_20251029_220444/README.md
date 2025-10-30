# MAE Reconstruction Demo Results

Generated: 2025-10-29 22:06:12

## Model Information
- Model: MAE ViT-Large with Complete Pretrained Weights
- Encoder: Pretrained on ImageNet
- Decoder: Pretrained for visualization
- Test Mask Ratios: 50%, 75%, 90%

## Reconstruction Results

### Image 1

| Mask Ratio | Loss | Actual Mask% | Mean Error | Max Error |
|------------|------|--------------|------------|----------|
| 50% | 0.3007 | 50.0% | 0.1130 | 0.9046 |
| 75% | 0.6046 | 75.0% | 0.1223 | 0.9771 |
| 90% | 0.8376 | 90.3% | 0.1381 | 0.9532 |

### Image 2

| Mask Ratio | Loss | Actual Mask% | Mean Error | Max Error |
|------------|------|--------------|------------|----------|
| 50% | 0.2491 | 50.0% | 0.0954 | 0.9189 |
| 75% | 0.3887 | 75.0% | 0.1028 | 0.8725 |
| 90% | 0.6842 | 90.3% | 0.1350 | 0.7898 |

### Image 3

| Mask Ratio | Loss | Actual Mask% | Mean Error | Max Error |
|------------|------|--------------|------------|----------|
| 50% | 0.2026 | 50.0% | 0.0905 | 0.7251 |
| 75% | 0.3315 | 75.0% | 0.0903 | 0.7564 |
| 90% | 0.9631 | 90.3% | 0.1579 | 0.7899 |

## File Structure

```
mae_english_demo_YYYYMMDD_HHMMSS/
├── mae_reconstruction_comparison.png  # Overview comparison
├── quality_comparison.png            # Quality comparison
├── image_1/                          # Individual results
│   ├── original.png
│   ├── mask_50percent/
│   ├── mask_75percent/
│   └── mask_90percent/
├── reconstruction_report.json        # Detailed statistics
└── README.md                         # This report
```
