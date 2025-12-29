"""
Stub module for perceptual losses - not needed for inference.
"""

import torch.nn as nn

class PerceptualLoss(nn.Module):
    """Stub class - not used during inference."""
    def __init__(self, **kwargs):
        super().__init__()
        raise NotImplementedError("Training functions not available in inference-only build")

class ResNetPL(nn.Module):
    """Stub class - not used during inference."""
    def __init__(self, **kwargs):
        super().__init__()
        raise NotImplementedError("Training functions not available in inference-only build")
