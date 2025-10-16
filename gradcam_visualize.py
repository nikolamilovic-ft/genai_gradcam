#!/usr/bin/env python3
"""
GradCAM Visualization Script for AI-Generated Image Detection
Generates heatmaps showing which regions the model focuses on for classification
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from pathlib import Path
import argparse

# Import timm for model loading
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class GradCAM:
    """GradCAM implementation for CNNs"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class):
        """Generate GradCAM heatmap for target class"""
        # Forward pass
        output = self.model(input_image)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU to focus on positive influence
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.squeeze().cpu().numpy()


def load_model(checkpoint_path, num_classes=2):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")

    # Create model
    model = timm.create_model('resnet50', pretrained=False, num_classes=num_classes)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DDP training)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    print(f"Model loaded successfully!")
    if 'epoch' in checkpoint:
        print(f"Checkpoint from epoch {checkpoint['epoch']}")

    return model


def sample_images(data_root, num_per_category=4):
    """Sample images from train/val and ai/nature directories"""
    data_root = Path(data_root)

    samples = {
        'train_ai': [],
        'train_nature': [],
        'val_ai': [],
        'val_nature': []
    }

    # Sample from each category
    for split in ['train', 'val']:
        for category in ['ai', 'nature']:
            category_path = data_root / split / category

            if not category_path.exists():
                print(f"Warning: {category_path} does not exist!")
                continue

            # Get all image files
            image_files = list(category_path.glob('*.jpg')) + \
                         list(category_path.glob('*.png')) + \
                         list(category_path.glob('*.JPEG'))

            image_files += list(category_path.glob("000_sdv4_00045.png"))

            if len(image_files) == 0:
                print(f"Warning: No images found in {category_path}")
                continue

            # Random sample
            sampled = random.sample(image_files, min(num_per_category, len(image_files)))
            # # include 000_sdv4_00045.png if exists
            # special_image = category_path / "000_sdv4_00045.png"
            # if special_image.exists() and special_image not in sampled:
            #     print("EXISTS************************************************")
            #     sampled[0] = special_image  # replace first image with special image
            samples[f"{split}_{category}"] = sampled
            print(f"Sampled {len(sampled)} images from {split}/{category}")

    return samples


def visualize_gradcam(image_path, model, transform, device, class_names, output_dir):
    """Generate and save GradCAM visualization for a single image"""

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    # Generate GradCAM
    # For ResNet50, use the last convolutional layer (layer4[-1])
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    cam = gradcam.generate_cam(input_tensor, predicted_class)

    # Resize CAM to match input image size
    cam_resized = np.array(Image.fromarray(cam).resize(img.size, Image.BILINEAR))

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    # GradCAM heatmap
    im = axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('GradCAM Heatmap', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(cam_resized, cmap='jet', alpha=0.5)
    axes[2].set_title(f'Overlay\nPrediction: {class_names[predicted_class]}\nConfidence: {confidence:.2%}',
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"{image_path.stem}_gradcam.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(description='Generate GradCAM visualizations for trained model')
    parser.add_argument('--checkpoint', type=str,
                       default='output/train/20251015-222347-resnet50-224/model_best.pth.tar',
                       help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str,
                       default='./data/',
                       help='Root directory of dataset (used in samples mode)')
    parser.add_argument('--image', type=str,
                       help='Path to a single image for direct mode (alternative to samples mode)')
    parser.add_argument('--output-dir', type=str, default='gradcam_outputs',
                       help='Directory to save GradCAM visualizations')
    parser.add_argument('--num-samples', type=int, default=4,
                       help='Number of samples per category (train_ai, train_nature, val_ai, val_nature)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load model
    model = load_model(args.checkpoint, num_classes=2)
    model = model.to(device)

    # Get data config and create transform
    data_config = resolve_data_config({}, model=model)
    transform = create_transform(**data_config, is_training=False)

    # Class names
    class_names = ['AI-Generated', 'Real (Nature)']

    # Check if running in direct mode (single image) or samples mode
    if args.image:
        # Direct mode: process single image
        print(f"\n{'='*60}")
        print("DIRECT MODE: Processing single image")
        print(f"{'='*60}")

        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            sys.exit(1)

        print(f"\nProcessing image: {image_path.name}")

        try:
            pred_class, confidence = visualize_gradcam(
                image_path, model, transform, device, class_names, output_dir
            )

            print(f"\n{'='*60}")
            print("RESULT")
            print(f"{'='*60}")
            print(f"Image: {image_path.name}")
            print(f"Prediction: {class_names[pred_class]}")
            print(f"Confidence: {confidence:.2%}")
            print(f"\n{'='*60}")
            print(f"Visualization saved to: {output_dir.absolute() / f'{image_path.stem}_gradcam.png'}")
            print(f"{'='*60}")

        except Exception as e:
            print(f"Error processing image: {e}")
            sys.exit(1)

    else:
        # Samples mode: process multiple images from dataset
        print(f"\n{'='*60}")
        print("SAMPLES MODE: Processing dataset samples")
        print(f"{'='*60}")

        print(f"\nSampling images from {args.data_root}")
        samples = sample_images(args.data_root, num_per_category=args.num_samples)

        # Generate GradCAM for all samples
        print(f"\nGenerating GradCAM visualizations...")
        results = {}

        for category, image_paths in samples.items():
            if len(image_paths) == 0:
                continue

            print(f"\nProcessing {category}:")
            results[category] = []

            category_dir = output_dir / category
            category_dir.mkdir(exist_ok=True)

            for img_path in image_paths:
                try:
                    pred_class, confidence = visualize_gradcam(
                        img_path, model, transform, device, class_names, category_dir
                    )
                    results[category].append({
                        'path': img_path,
                        'predicted': class_names[pred_class],
                        'confidence': confidence
                    })
                    print(f"  {img_path.name}: {class_names[pred_class]} ({confidence:.2%})")
                except Exception as e:
                    print(f"  Error processing {img_path.name}: {e}")

        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        for category, results_list in results.items():
            if len(results_list) == 0:
                continue

            true_label = 'AI-Generated' if 'ai' in category else 'Real (Nature)'
            correct = sum(1 for r in results_list if r['predicted'] == true_label)
            total = len(results_list)
            accuracy = correct / total if total > 0 else 0

            print(f"\n{category.upper().replace('_', ' ')}:")
            print(f"  True Label: {true_label}")
            print(f"  Correct Predictions: {correct}/{total} ({accuracy:.1%})")
            avg_conf = np.mean([r['confidence'] for r in results_list])
            print(f"  Average Confidence: {avg_conf:.2%}")

        print(f"\n{'='*60}")
        print(f"All visualizations saved to: {output_dir.absolute()}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
