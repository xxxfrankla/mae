#!/usr/bin/env python3
"""
English MAE Reconstruction Demo with Image Saving
High-quality reconstruction using complete pretrained MAE model
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import models_mae
from animediffusion_dataset_loader import create_animediffusion_dataloader
from datetime import datetime

# Set English font and environment
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def load_complete_mae_model():
    """Load complete MAE model with decoder weights"""
    print("\n🎯 Loading Complete MAE Model with Decoder...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create ViT-Large model (matching visualization model)
    model = models_mae.mae_vit_large_patch16()
    
    # Load complete pretrained weights
    model_path = 'complete_mae_models/mae_visualize_vit_large.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please run: ./download_complete_mae.sh")
        return None, None
    
    try:
        print(f"📥 Loading complete model weights: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check model structure
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Count parameters
        encoder_keys = [k for k in state_dict.keys() if not k.startswith('decoder') and k != 'mask_token']
        decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder') or k == 'mask_token']
        
        print(f"  Encoder parameters: {len(encoder_keys)}")
        print(f"  Decoder parameters: {len(decoder_keys)}")
        
        # Load weights
        msg = model.load_state_dict(state_dict, strict=False)
        if msg.missing_keys:
            print(f"  ⚠️  Missing keys: {len(msg.missing_keys)}")
        if msg.unexpected_keys:
            print(f"  ⚠️  Unexpected keys: {len(msg.unexpected_keys)}")
        
        model = model.to(device)
        model.eval()
        
        print("✅ Complete MAE model loaded successfully!")
        return model, device
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None

def create_output_folder():
    """Create output folder for saving images"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"mae_english_demo_{timestamp}"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"✅ Created output folder: {folder_name}")
    
    return folder_name

def save_individual_images(images, masks, reconstructions, errors, output_folder, mask_ratios):
    """Save individual reconstruction images"""
    print("\n💾 Saving individual reconstruction images...")
    
    # Inverse normalization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    saved_files = []
    
    for img_idx in range(len(images)):
        img_folder = os.path.join(output_folder, f'image_{img_idx+1}')
        os.makedirs(img_folder, exist_ok=True)
        
        # Save original image
        original_display = torch.clamp(inv_normalize(images[img_idx]).cpu(), 0, 1)
        original_pil = transforms.ToPILImage()(original_display)
        original_path = os.path.join(img_folder, 'original.png')
        original_pil.save(original_path)
        saved_files.append(original_path)
        
        for mask_idx, mask_ratio in enumerate(mask_ratios):
            mask_folder = os.path.join(img_folder, f'mask_{int(mask_ratio*100)}percent')
            os.makedirs(mask_folder, exist_ok=True)
            
            # Save masked image
            mask_vis = masks[img_idx][mask_idx]
            masked_img = original_display * (1 - mask_vis.cpu()) + mask_vis.cpu() * 0.5
            masked_pil = transforms.ToPILImage()(masked_img)
            masked_path = os.path.join(mask_folder, 'masked.png')
            masked_pil.save(masked_path)
            saved_files.append(masked_path)
            
            # Save reconstructed image
            reconstructed_display = torch.clamp(inv_normalize(reconstructions[img_idx][mask_idx]).cpu(), 0, 1)
            reconstructed_pil = transforms.ToPILImage()(reconstructed_display)
            reconstructed_path = os.path.join(mask_folder, 'reconstructed.png')
            reconstructed_pil.save(reconstructed_path)
            saved_files.append(reconstructed_path)
            
            # Save error map
            error_display = errors[img_idx][mask_idx].mean(dim=0)
            error_normalized = (error_display - error_display.min()) / (error_display.max() - error_display.min() + 1e-8)
            error_pil = transforms.ToPILImage()(error_normalized)
            error_path = os.path.join(mask_folder, 'error_map.png')
            error_pil.save(error_path)
            saved_files.append(error_path)
    
    print(f"✅ Saved {len(saved_files)} individual images")
    return saved_files

def create_english_reconstruction_demo(model, device, output_folder):
    """Create English reconstruction demo"""
    print("\n🎨 Creating English Reconstruction Demo...")
    
    # Try to load AnimeDiffusion data
    try:
        dataloader, dataset = create_animediffusion_dataloader(
            batch_size=3,
            max_samples=10,
            input_size=224,
            num_workers=0
        )
        
        if dataloader is not None:
            for images, _ in dataloader:
                test_images = images[:3]
                print(f"✅ Using AnimeDiffusion data: {test_images.shape}")
                break
        else:
            raise Exception("Dataloader creation failed")
            
    except Exception as e:
        print(f"⚠️  AnimeDiffusion loading failed: {e}")
        print("🎨 Using fallback test images...")
        test_images = create_test_images(device)
    
    # Inverse normalization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    test_images = test_images.to(device)
    num_images = test_images.shape[0]
    
    # Test different mask ratios
    mask_ratios = [0.5, 0.75, 0.9]
    
    # Store results for saving
    all_images = []
    all_masks = []
    all_reconstructions = []
    all_errors = []
    
    # Create visualization
    fig, axes = plt.subplots(num_images, len(mask_ratios)*3 + 1, figsize=(20, num_images*4))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    reconstruction_stats = []
    
    for img_idx in range(num_images):
        img_tensor = test_images[img_idx:img_idx+1]
        
        print(f"\n  🔍 Processing Image {img_idx+1}...")
        
        # Display original image
        original_display = torch.clamp(inv_normalize(img_tensor[0]).cpu(), 0, 1)
        axes[img_idx, 0].imshow(original_display.permute(1, 2, 0))
        axes[img_idx, 0].set_title(f'Original Image {img_idx+1}')
        axes[img_idx, 0].axis('off')
        
        all_images.append(img_tensor[0])
        
        col_idx = 1
        img_stats = {'image_id': img_idx + 1, 'results': []}
        
        img_masks = []
        img_reconstructions = []
        img_errors = []
        
        for mask_ratio in mask_ratios:
            print(f"    🎭 Mask Ratio: {mask_ratio*100:.0f}%")
            
            with torch.no_grad():
                # MAE forward pass
                loss, pred, mask = model(img_tensor, mask_ratio=mask_ratio)
                
                # Reconstruct image
                reconstructed = model.unpatchify(pred)
                
                # Create mask visualization
                mask_vis = mask.detach()
                mask_vis = mask_vis.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
                mask_vis = model.unpatchify(mask_vis)
                
                # Convert to display format
                reconstructed_display = torch.clamp(inv_normalize(reconstructed[0]).cpu(), 0, 1)
                masked_img = original_display * (1 - mask_vis[0].cpu()) + mask_vis[0].cpu() * 0.5
                
                # Calculate reconstruction error
                error = torch.abs(original_display - reconstructed_display)
                
                # Store for saving
                img_masks.append(mask_vis[0])
                img_reconstructions.append(reconstructed[0])
                img_errors.append(error)
                
                # Display results
                axes[img_idx, col_idx].imshow(masked_img.permute(1, 2, 0))
                axes[img_idx, col_idx].set_title(f'Masked {mask_ratio*100:.0f}%')
                axes[img_idx, col_idx].axis('off')
                
                axes[img_idx, col_idx+1].imshow(reconstructed_display.permute(1, 2, 0))
                axes[img_idx, col_idx+1].set_title(f'Reconstructed\nLoss: {loss.item():.3f}')
                axes[img_idx, col_idx+1].axis('off')
                
                error_display = error.mean(dim=0)
                im = axes[img_idx, col_idx+2].imshow(error_display, cmap='hot')
                axes[img_idx, col_idx+2].set_title(f'Error Map\nMean: {error.mean():.3f}')
                axes[img_idx, col_idx+2].axis('off')
                plt.colorbar(im, ax=axes[img_idx, col_idx+2], fraction=0.046, pad=0.04)
                
                col_idx += 3
                
                # Record statistics
                stats = {
                    'mask_ratio': mask_ratio,
                    'loss': loss.item(),
                    'actual_mask_ratio': mask.float().mean().item(),
                    'mean_error': error.mean().item(),
                    'max_error': error.max().item(),
                    'pred_range': [pred.min().item(), pred.max().item()]
                }
                
                img_stats['results'].append(stats)
                
                print(f"      Loss: {loss.item():.4f}")
                print(f"      Pred Range: [{pred.min():.3f}, {pred.max():.3f}]")
                print(f"      Reconstruction Error: {error.mean():.4f}")
        
        all_masks.append(img_masks)
        all_reconstructions.append(img_reconstructions)
        all_errors.append(img_errors)
        reconstruction_stats.append(img_stats)
    
    plt.tight_layout()
    
    # Save comparison visualization
    comparison_path = os.path.join(output_folder, 'mae_reconstruction_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison visualization saved: {comparison_path}")
    
    # Save individual images
    saved_files = save_individual_images(all_images, all_masks, all_reconstructions, all_errors, output_folder, mask_ratios)
    
    try:
        plt.show()
    except:
        print("💡 Run in GUI environment to display images")
    
    return comparison_path, saved_files, reconstruction_stats

def create_test_images(device):
    """Create test images"""
    images = []
    
    # Image 1: Clear geometric pattern
    img1 = torch.zeros(3, 224, 224, device=device)
    
    # Background gradient
    for i in range(224):
        for j in range(224):
            img1[0, i, j] = (0.2 + 0.3 * i / 224 - 0.485) / 0.229
            img1[1, i, j] = (0.3 + 0.4 * j / 224 - 0.456) / 0.224
            img1[2, i, j] = (0.6 - 0.406) / 0.225
    
    # Add circle
    y, x = torch.meshgrid(torch.arange(224, device=device), torch.arange(224, device=device), indexing='ij')
    circle = (x - 112)**2 + (y - 112)**2 <= 40**2
    img1[0][circle] = (0.9 - 0.485) / 0.229
    img1[1][circle] = (0.1 - 0.456) / 0.224
    img1[2][circle] = (0.1 - 0.406) / 0.225
    
    images.append(img1)
    
    # Image 2: Checkerboard
    img2 = torch.zeros(3, 224, 224, device=device)
    for i in range(0, 224, 32):
        for j in range(0, 224, 32):
            if (i//32 + j//32) % 2 == 0:
                color = [(0.8 - 0.485) / 0.229, (0.8 - 0.456) / 0.224, (0.8 - 0.406) / 0.225]
            else:
                color = [(0.2 - 0.485) / 0.229, (0.2 - 0.456) / 0.224, (0.2 - 0.406) / 0.225]
            
            img2[:, i:i+32, j:j+32] = torch.tensor(color, device=device).reshape(3, 1, 1)
    
    images.append(img2)
    
    # Image 3: Concentric circles
    img3 = torch.zeros(3, 224, 224, device=device)
    for r in range(20, 100, 20):
        mask = ((x - 112)**2 + (y - 112)**2 >= (r-10)**2) & ((x - 112)**2 + (y - 112)**2 <= r**2)
        color_intensity = r / 100
        img3[0][mask] = (color_intensity - 0.485) / 0.229
        img3[1][mask] = (0.5 - 0.456) / 0.224
        img3[2][mask] = (1.0 - color_intensity - 0.406) / 0.225
    
    images.append(img3)
    
    return torch.stack(images)

def create_quality_comparison(model, device, output_folder):
    """Create quality comparison with random decoder"""
    print("\n📊 Creating Quality Comparison...")
    
    # Load complete model (already loaded)
    complete_model = model
    
    # Create random decoder model
    random_model = models_mae.mae_vit_large_patch16()
    
    # Only load encoder weights
    model_path = 'complete_mae_models/mae_visualize_vit_large.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model']
    
    encoder_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith('decoder') and key != 'mask_token':
            encoder_state_dict[key] = value
    
    random_model.load_state_dict(encoder_state_dict, strict=False)
    random_model = random_model.to(device)
    random_model.eval()
    
    # Create test image
    test_img = create_test_images(device)[0:1]  # Only use first image
    
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    models_to_test = [
        ("Complete Pretrained Model", complete_model),
        ("Random Decoder Model", random_model)
    ]
    
    for i, (model_name, test_model) in enumerate(models_to_test):
        print(f"  🔍 Testing {model_name}...")
        
        with torch.no_grad():
            # Original image
            original_display = torch.clamp(inv_normalize(test_img[0]).cpu(), 0, 1)
            axes[i, 0].imshow(original_display.permute(1, 2, 0))
            axes[i, 0].set_title(f'{model_name}\nOriginal Image')
            axes[i, 0].axis('off')
            
            # MAE reconstruction
            loss, pred, mask = test_model(test_img, mask_ratio=0.75)
            reconstructed = test_model.unpatchify(pred)
            
            # Masked image
            mask_vis = mask.detach()
            mask_vis = mask_vis.unsqueeze(-1).repeat(1, 1, test_model.patch_embed.patch_size[0]**2 * 3)
            mask_vis = test_model.unpatchify(mask_vis)
            masked_img = original_display * (1 - mask_vis[0].cpu()) + mask_vis[0].cpu() * 0.5
            
            axes[i, 1].imshow(masked_img.permute(1, 2, 0))
            axes[i, 1].set_title('Masked Image (75%)')
            axes[i, 1].axis('off')
            
            # Reconstructed image
            reconstructed_display = torch.clamp(inv_normalize(reconstructed[0]).cpu(), 0, 1)
            axes[i, 2].imshow(reconstructed_display.permute(1, 2, 0))
            axes[i, 2].set_title(f'Reconstructed\nLoss: {loss.item():.3f}')
            axes[i, 2].axis('off')
            
            # Reconstruction error
            error = torch.abs(original_display - reconstructed_display)
            error_display = error.mean(dim=0)
            im = axes[i, 3].imshow(error_display, cmap='hot')
            axes[i, 3].set_title(f'Error Map\nMean: {error.mean():.3f}')
            axes[i, 3].axis('off')
            plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
            
            print(f"    Loss: {loss.item():.4f}")
            print(f"    Pred Range: [{pred.min():.3f}, {pred.max():.3f}]")
            print(f"    Reconstruction Error: {error.mean():.4f}")
    
    plt.tight_layout()
    
    comparison_path = os.path.join(output_folder, 'quality_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Quality comparison saved: {comparison_path}")
    
    try:
        plt.show()
    except:
        print("💡 Run in GUI environment to display images")
    
    return comparison_path

def save_reconstruction_report(stats, output_folder):
    """Save reconstruction report"""
    print("\n📄 Saving Reconstruction Report...")
    
    # Save JSON report
    import json
    json_path = os.path.join(output_folder, 'reconstruction_report.json')
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Generate Markdown report
    md_path = os.path.join(output_folder, 'README.md')
    with open(md_path, 'w') as f:
        f.write("# MAE Reconstruction Demo Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Information\n")
        f.write("- Model: MAE ViT-Large with Complete Pretrained Weights\n")
        f.write("- Encoder: Pretrained on ImageNet\n")
        f.write("- Decoder: Pretrained for visualization\n")
        f.write("- Test Mask Ratios: 50%, 75%, 90%\n\n")
        
        f.write("## Reconstruction Results\n\n")
        
        for img_result in stats:
            img_id = img_result['image_id']
            f.write(f"### Image {img_id}\n\n")
            
            f.write("| Mask Ratio | Loss | Actual Mask% | Mean Error | Max Error |\n")
            f.write("|------------|------|--------------|------------|----------|\n")
            
            for result in img_result['results']:
                f.write(f"| {result['mask_ratio']*100:.0f}% | "
                       f"{result['loss']:.4f} | "
                       f"{result['actual_mask_ratio']*100:.1f}% | "
                       f"{result['mean_error']:.4f} | "
                       f"{result['max_error']:.4f} |\n")
            
            f.write("\n")
        
        f.write("## File Structure\n\n")
        f.write("```\n")
        f.write("mae_english_demo_YYYYMMDD_HHMMSS/\n")
        f.write("├── mae_reconstruction_comparison.png  # Overview comparison\n")
        f.write("├── quality_comparison.png            # Quality comparison\n")
        f.write("├── image_1/                          # Individual results\n")
        f.write("│   ├── original.png\n")
        f.write("│   ├── mask_50percent/\n")
        f.write("│   ├── mask_75percent/\n")
        f.write("│   └── mask_90percent/\n")
        f.write("├── reconstruction_report.json        # Detailed statistics\n")
        f.write("└── README.md                         # This report\n")
        f.write("```\n")
    
    print(f"✅ Reports saved:")
    print(f"   JSON: {json_path}")
    print(f"   Markdown: {md_path}")
    
    return json_path, md_path

def main():
    """Main function"""
    print("🎯 English MAE Reconstruction Demo with Image Saving")
    print("=" * 60)
    
    # Create output folder
    output_folder = create_output_folder()
    
    # Load complete model
    model, device = load_complete_mae_model()
    if model is None:
        return
    
    # Create reconstruction demo
    comparison_path, saved_files, stats = create_english_reconstruction_demo(model, device, output_folder)
    
    # Create quality comparison
    quality_comparison_path = create_quality_comparison(model, device, output_folder)
    
    # Save reports
    json_path, md_path = save_reconstruction_report(stats, output_folder)
    
    print(f"\n🎉 Demo Complete!")
    print(f"📁 Output Folder: {output_folder}")
    print(f"📊 Comparison: {comparison_path}")
    print(f"🔍 Quality Analysis: {quality_comparison_path}")
    print(f"📄 Report: {md_path}")
    print(f"💾 Individual Images: {len(saved_files)} files saved")
    
    print(f"\n💡 Key Findings:")
    print("✅ Complete pretrained model provides high-quality reconstruction")
    print("🎨 Decoder weights are crucial for reconstruction quality")
    print("📈 Pretrained decoder understands encoder features correctly")
    print("🔥 This is the correct way for high-quality MAE reconstruction!")

if __name__ == "__main__":
    main()

