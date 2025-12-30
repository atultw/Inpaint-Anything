"""
Example usage of the LAMA inpainting function.
"""
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
