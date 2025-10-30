#!/usr/bin/env python3
"""
Improved MAE Visualization Demo
Based on Facebook's original notebook but with complete pretrained model
"""

import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

# Set environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Check whether run in Colab
if 'google.colab' in sys.modules:
    print('Running in Colab.')
    os.system('pip3 install timm==0.4.12')  # Compatible version
    os.system('git clone https://github.com/facebookresearch/mae.git')
    sys.path.append('./mae')
else:
    print('Running in local environment.')
    sys.path.append('..')

# Import models
try:
    import models_mae
    print('✅ MAE models imported successfully')
except ImportError as e:
    print(f'❌ Failed to import MAE models: {e}')
    sys.exit(1)

# Define utility functions
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    """Display image with proper normalization"""
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_complete_model(chkpt_dir, arch='mae_vit_large_patch16'):
    """Load complete MAE model with decoder weights"""
    print(f'🤖 Loading complete model: {arch}')
    print(f'📥 Checkpoint: {chkpt_dir}')
    
    # Build model
    model = getattr(models_mae, arch)()
    
    # Load model
    if os.path.exists(chkpt_dir):
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        
        # Check what's in the checkpoint
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Count parameters
        encoder_keys = [k for k in state_dict.keys() if not k.startswith('decoder') and k != 'mask_token']
        decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder') or k == 'mask_token']
        
        print(f'  Encoder parameters: {len(encoder_keys)}')
        print(f'  Decoder parameters: {len(decoder_keys)}')
        
        if len(decoder_keys) > 0:
            print('  ✅ Complete model with decoder weights!')
        else:
            print('  ⚠️  Encoder-only model (decoder will be random)')
        
        msg = model.load_state_dict(state_dict, strict=False)
        print(f'  Loading result: {len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected')
        
        # Print model statistics
        total_params = sum(p.numel() for p in model.parameters())
        encoder_params = sum(p.numel() for name, p in model.named_parameters() 
                           if not name.startswith('decoder') and name != 'mask_token')
        decoder_params = sum(p.numel() for name, p in model.named_parameters() 
                           if name.startswith('decoder') or name == 'mask_token')
        
        print(f'  📊 Model Statistics:')
        print(f'    Total: {total_params:,} ({total_params/1e6:.1f}M)')
        print(f'    Encoder: {encoder_params:,} ({encoder_params/1e6:.1f}M)')
        print(f'    Decoder: {decoder_params:,} ({decoder_params/1e6:.1f}M)')
        
    else:
        print(f'❌ Checkpoint not found: {chkpt_dir}')
        return None
    
    return model

def run_one_image(img, model, mask_ratio=0.75):
    """Run MAE on one image with improved error handling"""
    try:
        x = torch.tensor(img)

        # Make it a batch-like
        x = x.unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)

        # Run MAE
        with torch.no_grad():
            loss, y, mask = model(x.float(), mask_ratio=mask_ratio)
            y = model.unpatchify(y)
            y = torch.einsum('nchw->nhwc', y).detach().cpu()

        # Visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        
        x = torch.einsum('nchw->nhwc', x)

        # Masked image
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        return x, im_masked, y, im_paste, loss, mask
    
    except Exception as e:
        print(f'❌ Error in reconstruction: {e}')
        return None, None, None, None, None, None

def download_complete_model():
    """Download complete MAE model if not exists"""
    model_dir = 'complete_mae_models'
    os.makedirs(model_dir, exist_ok=True)
    
    model_url = 'https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth'
    model_path = os.path.join(model_dir, 'mae_visualize_vit_large.pth')
    
    if not os.path.exists(model_path):
        print('📥 Downloading complete MAE model with decoder...')
        print(f'URL: {model_url}')
        print('This may take a few minutes (1.2GB file)...')
        
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f'\rProgress: {progress:.1f}% ({downloaded/1024/1024:.1f}MB/{total_size/1024/1024:.1f}MB)', end='')
            
            print('\n✅ Download complete!')
        except Exception as e:
            print(f'\n❌ Download failed: {e}')
            return None
    else:
        print('✅ Complete MAE model already exists')
    
    return model_path

def main():
    """Main demonstration function"""
    print("🎭 Improved MAE Visualization Demo")
    print("=" * 50)
    
    # Download complete model
    model_path = download_complete_model()
    if model_path is None:
        return
    
    # Load complete MAE model
    model_mae = prepare_complete_model(model_path, 'mae_vit_large_patch16')
    
    if model_mae is None:
        print("❌ Failed to load model")
        return
    
    model_mae.eval()
    
    # Load test image
    print("\n📷 Loading test image...")
    img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg'  # fox
    
    try:
        img = Image.open(requests.get(img_url, stream=True).raw)
        img = img.resize((224, 224))
        img = np.array(img) / 255.
        
        assert img.shape == (224, 224, 3)
        
        # Normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std
        
        print("✅ Test image loaded and preprocessed")
        
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return
    
    # Test different mask ratios
    mask_ratios = [0.5, 0.75, 0.9]
    
    print(f"\n🎨 Testing reconstruction with different mask ratios...")
    
    for i, mask_ratio in enumerate(mask_ratios):
        print(f"\n--- Mask Ratio: {mask_ratio*100:.0f}% ---")
        
        # Set seed for reproducible results
        torch.manual_seed(2 + i)
        
        # Run reconstruction
        results = run_one_image(img, model_mae, mask_ratio=mask_ratio)
        x, im_masked, y, im_paste, loss, mask = results
        
        if x is not None:
            # Create visualization
            plt.figure(figsize=(20, 5))
            
            plt.subplot(1, 4, 1)
            show_image(x[0], "Original")
            
            plt.subplot(1, 4, 2)
            show_image(im_masked[0], f"Masked ({mask_ratio*100:.0f}%)")
            
            plt.subplot(1, 4, 3)
            show_image(y[0], f"Reconstruction\nLoss: {loss.item():.3f}")
            
            plt.subplot(1, 4, 4)
            show_image(im_paste[0], "Reconstruction + Visible")
            
            plt.tight_layout()
            
            # Save figure
            output_path = f'mae_reconstruction_mask_{int(mask_ratio*100)}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {output_path}")
            
            plt.show()
            
            # Print statistics
            actual_mask_ratio = mask.float().mean().item()
            print(f"📊 Statistics:")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Actual mask ratio: {actual_mask_ratio:.2%}")
            print(f"  Target mask ratio: {mask_ratio:.2%}")
        else:
            print("❌ Reconstruction failed")
    
    print(f"\n🎉 Demo completed!")
    print(f"💡 Key improvements over original:")
    print(f"  ✅ Uses complete model with decoder weights")
    print(f"  ✅ Better reconstruction quality")
    print(f"  ✅ Automatic model download")
    print(f"  ✅ Multiple mask ratio comparison")
    print(f"  ✅ Improved error handling")

if __name__ == "__main__":
    main()
