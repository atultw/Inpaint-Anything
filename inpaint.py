"""
Minimal inpainting function using LaMa (big-lama) model.

This module provides a simple function to remove objects from 2D images
given an image and a binary mask.
"""

import os
import yaml
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf

# Suppress OpenMP threading warnings
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Add lama to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "lama"))

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo


def remove_object(
    image: np.ndarray,
    mask: np.ndarray,
    lama_config: str = "./lama/configs/prediction/default.yaml",
    lama_ckpt: str = "./pretrained_models/big-lama",
    device: str = "cuda"
) -> np.ndarray:
    """
    Remove an object from a 2D image using the LaMa inpainting model.
    
    Args:
        image: Input image as numpy array with shape (H, W, 3) in RGB format, 
               with values in range [0, 255], dtype uint8
        mask: Binary mask as numpy array with shape (H, W), where 1/255 indicates 
              the region to inpaint and 0 indicates the region to keep
        lama_config: Path to LaMa model config file (default: ./lama/configs/prediction/default.yaml)
        lama_ckpt: Path to LaMa checkpoint directory (default: ./pretrained_models/big-lama)
        device: Device to run inference on, "cuda" or "cpu" (default: "cuda")
    
    Returns:
        Inpainted image as numpy array with shape (H, W, 3) in RGB format,
        with values in range [0, 255], dtype uint8
    
    Example:
        >>> import numpy as np
        >>> from PIL import Image
        >>> 
        >>> # Load image and mask
        >>> image = np.array(Image.open("image.jpg"))
        >>> mask = np.array(Image.open("mask.png").convert("L"))
        >>> 
        >>> # Remove object
        >>> result = remove_object(image, mask)
        >>> 
        >>> # Save result
        >>> Image.fromarray(result).save("result.jpg")
    """
    # Validate inputs
    if not isinstance(image, np.ndarray) or len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be a numpy array with shape (H, W, 3)")
    
    if not isinstance(mask, np.ndarray) or len(mask.shape) != 2:
        raise ValueError("Mask must be a numpy array with shape (H, W)")
    
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image and mask must have same height and width. "
                        f"Got image {image.shape[:2]} and mask {mask.shape}")
    
    # Normalize mask to [0, 255]
    if np.max(mask) == 1:
        mask = mask * 255
    
    # Convert to tensors and normalize
    img_tensor = torch.from_numpy(image).float().div(255.)
    mask_tensor = torch.from_numpy(mask).float()
    
    # Load model configuration
    predict_config = OmegaConf.load(lama_config)
    predict_config.model.path = lama_ckpt
    device = torch.device(device)
    
    train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'
    
    # Load model checkpoint
    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)
    
    # Prepare batch
    batch = {}
    batch['image'] = img_tensor.permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = mask_tensor[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    
    # Pad to modulo for model processing
    mod = 8
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1
    
    # Run inference
    with torch.no_grad():
        batch = model(batch)
        result = batch[predict_config.out_key][0].permute(1, 2, 0)
        result = result.detach().cpu().numpy()
    
    # Unpad to original size
    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        result = result[:orig_height, :orig_width]
    
    # Convert back to uint8 [0, 255]
    result = np.clip(result * 255, 0, 255).astype('uint8')
    
    return result


if __name__ == "__main__":
    """
    Example usage:
    python inpaint.py
    """
    from PIL import Image
    
    # Check if example files exist
    example_image = "./example/remove-anything/dog.jpg"
    example_mask = "./example/remove-anything/dog/mask_0.png"
    
    if not os.path.exists(example_image) or not os.path.exists(example_mask):
        print("Example files not found. Please provide your own image and mask.")
        print("Usage:")
        print("  from inpaint import remove_object")
        print("  import numpy as np")
        print("  from PIL import Image")
        print("  ")
        print("  image = np.array(Image.open('your_image.jpg'))")
        print("  mask = np.array(Image.open('your_mask.png').convert('L'))")
        print("  result = remove_object(image, mask)")
        print("  Image.fromarray(result).save('result.jpg')")
    else:
        # Load example image and mask
        image = np.array(Image.open(example_image))
        mask = np.array(Image.open(example_mask).convert("L"))
        
        # Remove object
        print("Processing image...")
        result = remove_object(image, mask)
        
        # Save result
        output_path = "./inpainted_result.png"
        Image.fromarray(result).save(output_path)
        print(f"Result saved to {output_path}")
