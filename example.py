"""
Example usage of the inpaint.remove_object function.

This script demonstrates how to use the minimal inpainting function
to remove objects from images using the LaMa (big-lama) model.
"""

import numpy as np
from PIL import Image
from inpaint import remove_object

def example_with_circular_mask():
    """Example: Remove a circular region from an image."""
    print("Example 1: Remove a circular region from an image")
    print("="*60)
    
    # Create a sample image (you would load your own image here)
    # For demonstration, create a 512x512 image with colored rectangles
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    image[100:200, 100:200] = [255, 0, 0]  # Red square
    image[300:400, 300:400] = [0, 255, 0]  # Green square
    image[50:150, 350:450] = [0, 0, 255]   # Blue square
    
    # Create a circular mask
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (150, 150)  # Center on the red square
    radius = 75
    
    y, x = np.ogrid[:h, :w]
    mask_area = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    mask[mask_area] = 255
    
    # Save input image and mask for reference
    Image.fromarray(image).save("input_image.png")
    Image.fromarray(mask).save("input_mask.png")
    print("Created input_image.png and input_mask.png")
    
    # Remove the object (this will fail without the model checkpoint)
    try:
        # Device is auto-selected (CUDA if available, otherwise CPU)
        result = remove_object(image, mask)
        Image.fromarray(result).save("output_image.png")
        print("✓ Successfully removed object!")
        print("✓ Saved result to output_image.png")
    except FileNotFoundError as e:
        print(f"\n✗ Model checkpoint not found: {e}")
        print("\nTo run this example, you need to:")
        print("1. Download big-lama model from:")
        print("   https://disk.yandex.ru/d/ouP6l8VJ0HpMZg")
        print("2. Extract and place it at:")
        print("   ./pretrained_models/big-lama")
        print("\nThe directory structure should be:")
        print("  ./pretrained_models/big-lama/")
        print("      config.yaml")
        print("      models/")
        print("          best.ckpt")


def example_with_loaded_image():
    """Example: Load an image and mask from files."""
    print("\nExample 2: Load image and mask from files")
    print("="*60)
    
    # Replace these with your own image and mask files
    image_path = "your_image.jpg"
    mask_path = "your_mask.png"
    
    print(f"To use this example:")
    print(f"1. Replace '{image_path}' with your image file")
    print(f"2. Replace '{mask_path}' with your mask file")
    print(f"3. Make sure the mask is a grayscale image where")
    print(f"   white (255) indicates regions to inpaint")
    print(f"   and black (0) indicates regions to keep")
    
    # Uncomment and modify when you have your own images:
    # image = np.array(Image.open(image_path))
    # mask = np.array(Image.open(mask_path).convert("L"))
    # result = remove_object(image, mask)
    # Image.fromarray(result).save("result.jpg")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LaMa Inpainting - Example Usage")
    print("="*60 + "\n")
    
    # Run example 1
    example_with_circular_mask()
    
    # Show example 2
    example_with_loaded_image()
    
    print("\n" + "="*60)
    print("For more information, see README.md")
    print("="*60 + "\n")
