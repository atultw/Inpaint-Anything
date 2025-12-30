"""
Simple LAMA inpainting for object removal.
Provides a single function to inpaint an image given a binary mask.
"""
import os
import sys
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, str(Path(__file__).resolve().parent / "lama"))
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo


@torch.no_grad()
def inpaint(
    img: np.ndarray,
    mask: np.ndarray,
    model_path: str = "big-lama",
    device: str = None
) -> np.ndarray:
    """
    Inpaint an image using LAMA model.
    
    Args:
        img: Input image as numpy array of shape (H, W, 3) with values in [0, 255]
        mask: Binary mask as numpy array of shape (H, W) with values 0 or 1 (or 0-255)
             1 (or 255) indicates areas to inpaint
        model_path: Path to the LAMA model directory (default: "big-lama")
        device: Device to run inference on ("cuda" or "cpu"). If None, auto-detect.
    
    Returns:
        Inpainted image as numpy array of shape (H, W, 3) with values in [0, 255]
    """
    # Validate inputs
    assert len(img.shape) == 3 and img.shape[2] == 3, "Image must be (H, W, 3)"
    assert len(mask.shape) == 2, "Mask must be (H, W)"
    assert img.shape[:2] == mask.shape, "Image and mask must have same height and width"
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # Normalize mask to 0-255 range
    if np.max(mask) == 1:
        mask = mask * 255
    
    # Convert to tensors
    img_tensor = torch.from_numpy(img).float().div(255.)
    mask_tensor = torch.from_numpy(mask).float()
    
    # Load config
    config_path = os.path.join(model_path, 'config.yaml')
    with open(config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'
    
    # Load checkpoint
    checkpoint_path = os.path.join(model_path, 'models', 'best.ckpt')
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)
    
    # Prepare batch
    batch = {}
    batch['image'] = img_tensor.permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = mask_tensor[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], 8)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], 8)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1
    
    # Run inference
    batch = model(batch)
    result = batch['inpainted'][0].permute(1, 2, 0)
    result = result.detach().cpu().numpy()
    
    # Unpad if necessary
    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        result = result[:orig_height, :orig_width]
    
    # Convert back to uint8
    result = np.clip(result * 255, 0, 255).astype('uint8')
    return result
