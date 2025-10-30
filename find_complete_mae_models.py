#!/usr/bin/env python3
"""
寻找和下载包含解码器权重的完整MAE模型
"""

import os
import sys
import requests
import torch
from urllib.parse import urlparse
import json

def check_facebook_mae_models():
    """检查Facebook官方MAE模型"""
    print("🔍 检查Facebook官方MAE模型...")
    
    # Facebook MAE官方模型链接
    official_models = {
        "mae_pretrain_vit_base": {
            "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
            "description": "ViT-Base预训练模型（只有编码器）"
        },
        "mae_pretrain_vit_large": {
            "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth", 
            "description": "ViT-Large预训练模型（只有编码器）"
        },
        "mae_pretrain_vit_huge": {
            "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth",
            "description": "ViT-Huge预训练模型（只有编码器）"
        },
        "mae_finetuned_vit_base": {
            "url": "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth",
            "description": "ViT-Base微调模型（用于分类，无解码器）"
        },
        "mae_visualize_vit_large": {
            "url": "https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth",
            "description": "ViT-Large可视化模型（可能包含解码器！）"
        },
        "mae_visualize_vit_large_ganloss": {
            "url": "https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth",
            "description": "ViT-Large+GAN损失可视化模型（可能包含解码器！）"
        }
    }
    
    print(f"📋 发现 {len(official_models)} 个官方模型:")
    for name, info in official_models.items():
        print(f"  • {name}: {info['description']}")
    
    return official_models

def check_model_contents(model_path):
    """检查模型文件内容"""
    print(f"\n🔍 检查模型内容: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 文件不存在: {model_path}")
        return None
    
    try:
        # 加载模型检查点
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 分析键值
        encoder_keys = []
        decoder_keys = []
        other_keys = []
        
        for key in state_dict.keys():
            if key.startswith('decoder') or key == 'mask_token':
                decoder_keys.append(key)
            elif any(key.startswith(prefix) for prefix in ['patch_embed', 'pos_embed', 'cls_token', 'blocks', 'norm']):
                encoder_keys.append(key)
            else:
                other_keys.append(key)
        
        print(f"  📊 模型分析:")
        print(f"    编码器参数: {len(encoder_keys)} 个")
        print(f"    解码器参数: {len(decoder_keys)} 个")
        print(f"    其他参数: {len(other_keys)} 个")
        
        if len(decoder_keys) > 0:
            print(f"  ✅ 包含解码器权重!")
            print(f"    解码器参数示例: {decoder_keys[:5]}")
            return True
        else:
            print(f"  ❌ 不包含解码器权重")
            return False
            
    except Exception as e:
        print(f"  ❌ 检查失败: {e}")
        return None

def download_model(url, filename):
    """下载模型文件"""
    print(f"\n📥 下载模型: {filename}")
    print(f"  URL: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r  进度: {progress:.1f}% ({downloaded/1024/1024:.1f}MB/{total_size/1024/1024:.1f}MB)", end='')
        
        print(f"\n  ✅ 下载完成: {filename}")
        return True
        
    except Exception as e:
        print(f"\n  ❌ 下载失败: {e}")
        return False

def search_huggingface_models():
    """搜索HuggingFace上的MAE模型"""
    print("\n🤗 搜索HuggingFace上的MAE模型...")
    
    # 一些可能包含完整MAE的HuggingFace模型
    hf_models = [
        {
            "name": "facebook/vit-mae-base",
            "description": "Facebook官方ViT-MAE模型",
            "url": "https://huggingface.co/facebook/vit-mae-base"
        },
        {
            "name": "facebook/vit-mae-large", 
            "description": "Facebook官方ViT-MAE Large模型",
            "url": "https://huggingface.co/facebook/vit-mae-large"
        },
        {
            "name": "facebook/vit-mae-huge",
            "description": "Facebook官方ViT-MAE Huge模型", 
            "url": "https://huggingface.co/facebook/vit-mae-huge"
        }
    ]
    
    print("📋 HuggingFace MAE模型:")
    for model in hf_models:
        print(f"  • {model['name']}: {model['description']}")
        print(f"    URL: {model['url']}")
    
    return hf_models

def download_huggingface_model():
    """下载HuggingFace模型的示例代码"""
    print("\n📝 HuggingFace模型下载示例:")
    
    code_example = '''
# 安装transformers库
pip install transformers

# Python代码示例
from transformers import ViTMAEModel, ViTMAEConfig

# 加载预训练模型（包含编码器和解码器）
model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

# 这个模型包含完整的编码器和解码器
print("编码器层数:", len(model.encoder.layer))
print("解码器层数:", len(model.decoder.layer))

# 使用模型进行重建
import torch
from PIL import Image
import requests
from transformers import ViTMAEImageProcessor

# 加载图像处理器
processor = ViTMAEImageProcessor.from_pretrained("facebook/vit-mae-base")

# 加载测试图像
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 预处理
inputs = processor(images=image, return_tensors="pt")

# 前向传播（包含重建）
with torch.no_grad():
    outputs = model(**inputs)
    
# 获取重建结果
reconstructed_pixel_values = outputs.logits
print("重建图像形状:", reconstructed_pixel_values.shape)
'''
    
    print(code_example)
    
    return code_example

def create_download_script():
    """创建下载脚本"""
    print("\n📝 创建完整模型下载脚本...")
    
    script_content = '''#!/bin/bash
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
'''
    
    with open('download_complete_mae.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('download_complete_mae.sh', 0o755)
    print("✅ 下载脚本创建完成: download_complete_mae.sh")

def create_model_checker():
    """创建模型检查脚本"""
    print("\n📝 创建模型检查脚本...")
    
    checker_content = '''#!/usr/bin/env python3
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
'''
    
    with open('check_model_contents.py', 'w') as f:
        f.write(checker_content)
    
    print("✅ 模型检查脚本创建完成: check_model_contents.py")

def main():
    """主函数"""
    print("🔍 寻找包含解码器权重的完整MAE模型")
    print("=" * 60)
    
    # 1. 检查Facebook官方模型
    official_models = check_facebook_mae_models()
    
    # 2. 检查现有模型
    print(f"\n🔍 检查现有模型...")
    existing_models = [
        "pretrained_models/mae_pretrain_vit_base.pth",
        "pretrained_models/mae_finetuned_vit_base.pth"
    ]
    
    for model_path in existing_models:
        check_model_contents(model_path)
    
    # 3. 搜索HuggingFace模型
    hf_models = search_huggingface_models()
    
    # 4. 提供下载示例
    download_huggingface_model()
    
    # 5. 创建下载脚本
    create_download_script()
    create_model_checker()
    
    print(f"\n🎯 推荐方案:")
    print("1. 尝试下载可视化模型（可能包含解码器）:")
    print("   ./download_complete_mae.sh")
    print()
    print("2. 使用HuggingFace transformers库:")
    print("   pip install transformers")
    print("   from transformers import ViTMAEModel")
    print("   model = ViTMAEModel.from_pretrained('facebook/vit-mae-base')")
    print()
    print("3. 检查模型内容:")
    print("   python check_model_contents.py <model_path>")
    
    print(f"\n💡 关于reshape错误:")
    print("这通常是因为预测输出的形状与期望不匹配")
    print("可能需要调整unpatchify函数或检查patch_size设置")

if __name__ == "__main__":
    main()
