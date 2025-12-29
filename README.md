# Inpaint Anything - Minimal Version

This is a minimal version of Inpaint Anything that provides a simple Python function to remove objects from 2D images using the LaMa (big-lama) inpainting model.

## Features

- Simple function interface: pass in an image and mask as numpy arrays, get back the inpainted image
- Uses the powerful LaMa (Large Mask Inpainting) model
- Minimal dependencies
- No GUI, no SAM segmentation, no video/3D processing - just pure inpainting

## Installation

Requires `python>=3.8`

```bash
pip install -r requirements.txt
```

## Model Setup

You need to download the **big-lama** model checkpoint and place it in the correct location:

1. Download big-lama from [here](https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) or from [Google Drive](https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A?usp=sharing)
2. Extract and place it at `./pretrained_models/big-lama`

The directory structure should look like:
```
./pretrained_models/big-lama/
    config.yaml
    models/
        best.ckpt
```

## Usage

### Quick Example

Run the included example script to see how the function works:

```bash
python example.py
```

This will create sample images and show you the expected output format.

### Basic Usage

```python
from inpaint import remove_object
import numpy as np
from PIL import Image

# Load your image and mask
image = np.array(Image.open("your_image.jpg"))
mask = np.array(Image.open("your_mask.png").convert("L"))

# Remove the object
result = remove_object(image, mask)

# Save the result
Image.fromarray(result).save("result.jpg")
```

### Function Signature

```python
def remove_object(
    image: np.ndarray,
    mask: np.ndarray,
    lama_config: str = "./lama/configs/prediction/default.yaml",
    lama_ckpt: str = "./pretrained_models/big-lama",
    device: str = "cuda"
) -> np.ndarray
```

**Parameters:**
- `image`: Input image as numpy array with shape `(H, W, 3)` in RGB format, values in range `[0, 255]`, dtype `uint8`
- `mask`: Binary mask as numpy array with shape `(H, W)`, where `1` or `255` indicates the region to inpaint and `0` indicates the region to keep
- `lama_config`: Path to LaMa model config file (default: `./lama/configs/prediction/default.yaml`)
- `lama_ckpt`: Path to LaMa checkpoint directory (default: `./pretrained_models/big-lama`)
- `device`: Device to run inference on, `"cuda"` or `"cpu"` (default: `"cuda"`)

**Returns:**
- Inpainted image as numpy array with shape `(H, W, 3)` in RGB format, values in range `[0, 255]`, dtype `uint8`

### Creating a Mask

You need to provide a binary mask where the white pixels (value 1 or 255) indicate the object to remove, and black pixels (value 0) indicate areas to keep unchanged.

You can create masks using:
- Image editing software (Photoshop, GIMP, etc.)
- Python libraries (OpenCV, scikit-image, etc.)
- Other segmentation tools (SAM, etc.)

Example of creating a simple circular mask:

```python
import numpy as np

# Create a circular mask
h, w = image.shape[:2]
mask = np.zeros((h, w), dtype=np.uint8)
center = (w // 2, h // 2)
radius = 100

y, x = np.ogrid[:h, :w]
mask_area = (x - center[0])**2 + (y - center[1])**2 <= radius**2
mask[mask_area] = 255
```

## Technical Details

This implementation uses:
- **LaMa (Large Mask Inpainting)**: A state-of-the-art inpainting model that can fill in large missing regions in images
- The `big-lama` checkpoint is trained on high-resolution images and produces high-quality results

## Citation

If you use this code in your research, please cite the original Inpaint Anything and LaMa papers:

```bibtex
@article{yu2023inpaint,
  title={Inpaint Anything: Segment Anything Meets Image Inpainting},
  author={Yu, Tao and Feng, Runseng and Feng, Ruoyu and Liu, Jinming and Jin, Xin and Zeng, Wenjun and Chen, Zhibo},
  journal={arXiv preprint arXiv:2304.06790},
  year={2023}
}
```

```bibtex
@article{suvorov2021resolution,
  title={Resolution-robust Large Mask Inpainting with Fourier Convolutions},
  author={Suvorov, Roman and others},
  journal={arXiv preprint arXiv:2109.07161},
  year={2021}
}
```

## License

See LICENSE file for details.

## Acknowledgments

- [LaMa](https://github.com/advimman/lama) - The inpainting model
- [Inpaint Anything](https://github.com/geekyutao/Inpaint-Anything) - Original repository
