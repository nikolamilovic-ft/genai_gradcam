# GradCAM Visualization for AI-Generated Image Detection

Generate GradCAM heatmaps to visualize which regions of an image the model focuses on for classification.

## Requirements

```bash
python3.10 -m venv venv
source venv/bin/activate
python3.10 -m pip install -r requirements.txt
```

## Usage

### Direct Mode (Single Image)

Process a single image:

```bash
python3.10 gradcam_visualize.py --image /path/to/image.jpg
```

### Samples Mode (Dataset)

Process random samples from a dataset:

```bash
python gradcam_visualize.py \
    --data-root ./data/ \
    --num-samples 4
```

Expected dataset structure:
```
data/
├── train/
│   ├── ai/
│   └── nature/
└── val/
    ├── ai/
    └── nature/
```

## Arguments

- `--checkpoint`: Path to model checkpoint (the best one is default) (default: `output/train/.../model_best.pth.tar`)
- `--image`: Path to single image for direct mode
- `--data-root`: Root directory of dataset for samples mode (default: `./data/`)
- `--output-dir`: Directory to save visualizations (default: `gradcam_outputs`)
- `--num-samples`: Number of samples per category in samples mode (default: 4)
- `--device`: Device to use - `cuda` or `cpu` (default: `cuda`)

## Output

The script generates three visualizations for each image:
1. Original image
2. GradCAM heatmap
3. Overlay with prediction and confidence

All outputs are saved as PNG files in the specified output directory.
