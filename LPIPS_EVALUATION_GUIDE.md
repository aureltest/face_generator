# Arc2Face LPIPS Evaluation Guide

This guide explains how to evaluate Arc2Face-generated face images using the Learned Perceptual Image Patch Similarity (LPIPS) metric.

## Overview

LPIPS is a perceptual similarity metric that measures how similar two images appear to human observers. Unlike pixel-based metrics (MSE, PSNR), LPIPS uses deep neural network features to capture perceptual differences, making it particularly suitable for evaluating face generation quality.

### Key Points:
- **Lower LPIPS = Better similarity** (0 = identical, typically 0-1 range)
- Uses pre-trained networks (AlexNet recommended) to extract perceptual features
- Aligns well with human perception of image similarity
- Ideal for evaluating ID-consistency in face generation

## Installation

```bash
# Install LPIPS and visualization dependencies
pip install lpips matplotlib seaborn tqdm
```

## Quick Start

```python
from arc2face_lpips_eval import Arc2FaceLPIPSEvaluator

# Initialize evaluator
evaluator = Arc2FaceLPIPSEvaluator(
    device="cuda",      # or "cpu"
    lpips_net="alex",   # Best for evaluation
    image_size=256      # Standard LPIPS size
)

# Evaluate single image
results = evaluator.evaluate_single_image(
    "path/to/original_face.jpg",
    num_generated=4
)

print(f"Mean LPIPS: {results['mean_lpips']:.4f}")
```

## Usage Examples

### 1. Single Image Evaluation

Evaluate how well Arc2Face preserves identity for a single face:

```python
results = evaluator.evaluate_single_image(
    "face.jpg",
    num_generated=4,
    generation_params={
        "num_inference_steps": 25,
        "guidance_scale": 3.0,
        "seed": 42
    }
)
```

### 2. Batch Evaluation

Evaluate multiple faces and get aggregate statistics:

```python
image_paths = ["face1.jpg", "face2.jpg", "face3.jpg"]

batch_results = evaluator.evaluate_dataset(
    image_paths,
    num_generated_per_image=4,
    save_results=True,
    output_dir="evaluation_results"
)

print(f"Overall mean LPIPS: {batch_results['overall_mean_lpips']:.4f}")
```

### 3. Custom Evaluation Workflow

For fine-grained control over the evaluation process:

```python
# Extract face embedding
face_emb = evaluator.extract_face_embedding("face.jpg")

# Generate images with custom parameters
images = evaluator.generate_images(
    face_emb,
    num_images=4,
    num_inference_steps=50,
    guidance_scale=5.0
)

# Compute LPIPS for each generated image
for img in images:
    score = evaluator.compute_lpips("face.jpg", img)
    print(f"LPIPS: {score:.4f}")
```

## Interpreting Results

### LPIPS Score Ranges

| LPIPS Range | Interpretation | Quality Assessment |
|-------------|----------------|-------------------|
| < 0.1 | Excellent | Nearly identical, exceptional ID preservation |
| 0.1 - 0.3 | Very Good | Strong similarity, good ID preservation |
| 0.3 - 0.5 | Good | Moderate similarity, acceptable for most use cases |
| 0.5 - 0.7 | Fair | Noticeable differences, ID somewhat preserved |
| > 0.7 | Poor | Significant differences, poor ID preservation |

### Statistical Analysis

The evaluator provides comprehensive statistics:

- **Mean LPIPS**: Average perceptual distance across all generations
- **Std LPIPS**: Consistency of generation quality
- **Min/Max LPIPS**: Best and worst case performance
- **Percentiles**: Distribution of scores (25th, 50th, 75th, 90th, 95th)

## Output Structure

### Evaluation Results

The evaluator saves results in JSON format:

```json
{
  "num_images_evaluated": 10,
  "total_generations": 40,
  "overall_mean_lpips": 0.4523,
  "overall_std_lpips": 0.0834,
  "percentiles": {
    "25th": 0.3921,
    "50th": 0.4486,
    "75th": 0.5102
  },
  "individual_results": [...]
}
```

### Generated Files

```
evaluation_results/
├── evaluation_results.json    # Detailed metrics
├── lpips_distribution.png     # Visual analysis
├── person1/
│   ├── generated_0.png
│   ├── generated_1.png
│   └── ...
└── person2/
    └── ...
```

## Advanced Configuration

### Generation Parameters

Optimize generation quality vs. speed:

```python
generation_params = {
    "num_inference_steps": 25,  # 10-50, higher = better quality
    "guidance_scale": 3.0,      # 1-7, controls adherence to face
    "seed": 42                  # For reproducibility
}
```

### LPIPS Network Options

- **"alex"** (recommended): Best balance of speed and accuracy
- **"vgg"**: More traditional perceptual loss
- **"squeeze"**: Faster but less accurate

### Image Preprocessing

The evaluator automatically:
1. Resizes images to 256×256 (configurable)
2. Normalizes to [-1, 1] range
3. Handles face detection and alignment

## Best Practices

1. **Use Consistent Image Sizes**: Resize both original and generated images to the same dimensions
2. **Face Alignment**: Ensure faces are properly aligned for fair comparison
3. **Multiple Generations**: Generate 4+ images per face for robust statistics
4. **Seed Control**: Use fixed seeds for reproducible evaluations
5. **Batch Processing**: Evaluate multiple faces for dataset-level insights

## Troubleshooting

### Common Issues

1. **"No face detected"**: Ensure image contains a clear, frontal face
2. **High LPIPS scores**: Check if faces are properly aligned
3. **CUDA errors**: Fall back to CPU or check GPU compatibility
4. **Memory issues**: Reduce batch size or image resolution

### Performance Tips

- Use GPU acceleration when available
- Process images in batches
- Lower `num_inference_steps` for faster evaluation
- Pre-extract face embeddings for repeated evaluations

## Integration with Existing Workflows

```python
# Example: Integrate with your face dataset
import pandas as pd

# Load your dataset
df = pd.read_csv("face_dataset.csv")
image_paths = df["image_path"].tolist()

# Evaluate
results = evaluator.evaluate_dataset(image_paths)

# Add scores to dataframe
df["mean_lpips"] = [r["mean_lpips"] for r in results["individual_results"]]
df.to_csv("face_dataset_with_lpips.csv")
```

## Citation

If you use this evaluation code in your research, please cite:

```bibtex
@inproceedings{zhang2018perceptual,
  title={The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A and Shechtman, Eli and Wang, Oliver},
  booktitle={CVPR},
  year={2018}
}
```

## Summary

The Arc2Face LPIPS evaluator provides a comprehensive framework for assessing the perceptual quality and ID-consistency of generated face images. By leveraging the LPIPS metric, you can quantitatively measure how well Arc2Face preserves facial identity across different generation parameters and conditions.