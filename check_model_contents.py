#!/usr/bin/env python3
"""
检查MAE模型是否包含解码器权重
"""

import torch
import sys
import os

def check_mae_model(model_path):
    """检查MAE模型内容"""
    print(f"🔍 检查模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 文件不存在")
        return False
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 统计参数
        encoder_keys = [k for k in state_dict.keys() if not k.startswith('decoder') and k != 'mask_token']
        decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder') or k == 'mask_token']
        
        print(f"  编码器参数: {len(encoder_keys)}")
        print(f"  解码器参数: {len(decoder_keys)}")
        
        if len(decoder_keys) > 0:
            print(f"  ✅ 包含解码器权重!")
            print(f"  解码器参数示例:")
            for key in decoder_keys[:10]:
                print(f"    - {key}")
            return True
        else:
            print(f"  ❌ 不包含解码器权重")
            return False
            
    except Exception as e:
        print(f"  ❌ 检查失败: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_model_contents.py <model_path>")
        print("示例: python check_model_contents.py pretrained_models/mae_pretrain_vit_base.pth")
        sys.exit(1)
    
    model_path = sys.argv[1]
    check_mae_model(model_path)
