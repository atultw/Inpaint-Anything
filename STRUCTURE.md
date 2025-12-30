# Repository Structure

This is a minimal LAMA inpainting implementation with only the essential code needed for object removal.

## File Structure

```
.
├── LICENSE                  # Original license
├── README.md               # Updated minimal documentation
├── requirements.txt        # Minimal dependencies (torch, numpy, PIL, etc.)
├── inpaint.py             # Main API - single inpaint() function
├── example.py             # Usage example
├── test_inpaint.py        # Test script
└── lama/                  # Minimal LAMA implementation
    ├── configs/
    │   └── prediction/
    │       └── default.yaml
    └── saicinpainting/
        ├── __init__.py
        ├── utils.py
        ├── evaluation/
        │   ├── __init__.py
        │   ├── data.py
        │   └── utils.py
        └── training/
            ├── __init__.py
            ├── data/
            │   ├── __init__.py
            │   └── datasets.py
            ├── losses/
            │   ├── __init__.py
            │   ├── adversarial.py
            │   ├── distance_weighting.py
            │   ├── feature_matching.py
            │   └── perceptual.py
            ├── modules/
            │   ├── __init__.py
            │   ├── base.py
            │   ├── depthwise_sep_conv.py
            │   ├── fake_fakes.py
            │   ├── ffc.py
            │   ├── multidilated_conv.py
            │   ├── spatial_transform.py
            │   └── squeeze_excitation.py
            ├── trainers/
            │   ├── __init__.py
            │   ├── base.py
            │   └── default.py
            └── visualizers/
                ├── __init__.py
                └── noop.py
```

## What Was Removed

### Complete Folders Removed:
- `segment_anything/` - SAM model (not needed for LAMA-only inpainting)
- `app/` - Web UI application
- `example/` - Example images and demos
- `nerf/` - 3D scene support
- `pytracking/` - Object tracking
- `sttn/` - Alternative video inpainting method
- `pretrained_models/` - Model weights (user must download separately)
- `weights/` - Additional model weights
- `script/` - Shell scripts for various features
- `utils/` - Various utility functions

### Python Files Removed:
- `remove_anything.py` - SAM-based object removal
- `remove_anything_3d.py` - 3D scene object removal
- `remove_anything_video.py` - Video object removal
- `replace_anything.py` - Object replacement
- `fill_anything.py` - Object filling with Stable Diffusion
- `sam_segment.py` - SAM segmentation interface
- `stable_diffusion_inpaint.py` - Stable Diffusion inpainting
- `sttn_video_inpaint.py` - Video inpainting with STTN
- `ostrack.py` - Object tracking
- `lama_inpaint.py` - Old LAMA interface (replaced with new inpaint.py)

### From lama/ folder:
- Training code and scripts
- Data generation utilities
- Evaluation metrics and losses (kept minimal stubs)
- Docker and conda configuration
- Binary files and documentation
- Most config files (kept only prediction/default.yaml)

## Dependencies

Only 7 packages are needed:
1. torch - PyTorch
2. torchvision - PyTorch vision utilities
3. numpy - Numerical arrays
4. pillow - Image I/O
5. opencv-python - Image processing
6. pyyaml - YAML config parsing
7. omegaconf - Advanced config handling

**Removed dependencies:**
- segment-anything, transformers, accelerate (for SAM/Stable Diffusion)
- diffusers, safetensors (for Stable Diffusion)
- scipy, scikit-image, scikit-learn (for evaluation)
- matplotlib, pandas (for visualization)
- albumentations, kornia (for data augmentation)
- pytorch-lightning (for training)
- tensorflow, timm (for various models)
- easydict, tabulate, packaging, webdataset, wldhx.yadisk-direct (utilities)

## API Usage

The entire API is a single function:

```python
from inpaint import inpaint

result = inpaint(img, mask, model_path='big-lama', device=None)
```

**Parameters:**
- `img`: numpy array (H, W, 3), values [0, 255]
- `mask`: numpy array (H, W), values 0/1 or 0/255, where 1/255 = inpaint
- `model_path`: path to big-lama model directory
- `device`: 'cuda', 'cpu', or None (auto-detect)

**Returns:**
- numpy array (H, W, 3), values [0, 255]

## File Count

- Original repository: ~500+ Python files
- Minimal version: ~26 Python files in lama/ + 3 main files
- Reduction: ~95% fewer files

## Size

- Original (excluding .git): ~200+ MB
- Minimal (excluding .git): ~100 KB Python code
- The big-lama model (not included): ~200 MB
