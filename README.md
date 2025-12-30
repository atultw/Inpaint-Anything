# LAMA Inpainting - Minimal Object Removal

A minimal, streamlined implementation for object removal from images using the LAMA (Large Mask Inpainting) model.

This repository provides a simple Python API for inpainting: given an image and a binary mask, it returns the inpainted result.

## Features

- ✅ Simple API: Single function call for inpainting
- ✅ Minimal dependencies
- ✅ Based on the powerful [LaMa](https://arxiv.org/abs/2109.07161) model
- ✅ Works with numpy arrays directly

## Installation

1. Clone this repository:
```bash
git clone https://github.com/atultw/Inpaint-Anything.git
cd Inpaint-Anything
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the LAMA model:
```bash
# Download the big-lama model (pretrained weights)
# Place the model in a directory named "big-lama" in the root of the repository
# The directory should contain:
#   - config.yaml
#   - models/best.ckpt
```

You can download the big-lama model from the [original LAMA repository](https://github.com/advimman/lama).

## Usage

```python
import numpy as np
from PIL import Image
from inpaint import inpaint

# Load image and mask
img = np.array(Image.open('input_image.jpg'))
mask = np.array(Image.open('mask.png').convert('L'))

# Inpaint (assumes big-lama model is in ./big-lama directory)
result = inpaint(img, mask, model_path='big-lama')

# Save result
Image.fromarray(result).save('output.jpg')
```

### Parameters

- `img` (np.ndarray): Input image of shape (H, W, 3) with values in [0, 255]
- `mask` (np.ndarray): Binary mask of shape (H, W) with values 0 or 1 (or 0-255)
  - 1 (or 255) indicates areas to inpaint
  - 0 indicates areas to keep unchanged
- `model_path` (str): Path to the LAMA model directory (default: "big-lama")
- `device` (str, optional): Device to run inference on ("cuda" or "cpu"). Auto-detects if not specified.

### Returns

- `np.ndarray`: Inpainted image of shape (H, W, 3) with values in [0, 255]

## Example

See `example.py` for a complete usage example.

## Credits

This is a minimal version derived from the original [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything) repository, focusing only on the LAMA inpainting functionality.

The LAMA model is from [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161).

## License

See LICENSE file for details.
