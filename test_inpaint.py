#!/usr/bin/env python3
"""
Test script for the minimal LAMA inpainting implementation.

This script tests the basic functionality of the inpaint function.
Note: Requires PyTorch and the big-lama model to be installed.
"""

import sys
import numpy as np
from PIL import Image
import os

def test_import():
    """Test that the inpaint module can be imported."""
    try:
        from inpaint import inpaint
        print("✓ Successfully imported inpaint module")
        return True
    except Exception as e:
        print(f"✗ Failed to import inpaint module: {e}")
        return False


def test_basic_functionality():
    """Test basic inpaint functionality with dummy data."""
    try:
        from inpaint import inpaint
        
        # Create a dummy image (100x100 RGB)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Create a dummy mask (100x100 binary)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255  # Mark center region for inpainting
        
        print("✓ Created test image and mask")
        print(f"  Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
        
        # Check if big-lama model exists
        if not os.path.exists('big-lama'):
            print("⚠ big-lama model not found. Please download it first.")
            print("  The test will not run the actual inpainting.")
            return True
        
        # Try to run inpainting
        print("Running inpainting...")
        result = inpaint(img, mask, model_path='big-lama')
        
        print(f"✓ Inpainting completed successfully")
        print(f"  Result shape: {result.shape}, dtype: {result.dtype}")
        
        # Verify result properties
        assert result.shape == img.shape, "Result shape doesn't match input"
        assert result.dtype == np.uint8, "Result dtype is not uint8"
        assert np.all((result >= 0) & (result <= 255)), "Result values out of range"
        
        print("✓ All assertions passed")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_image():
    """Test with a real image file if available."""
    try:
        from inpaint import inpaint
        
        # Check for test image
        test_image_path = 'test_image.jpg'
        test_mask_path = 'test_mask.png'
        
        if not os.path.exists(test_image_path) or not os.path.exists(test_mask_path):
            print("⚠ Test images not found. Skipping real image test.")
            return True
        
        print("Loading test image and mask...")
        img = np.array(Image.open(test_image_path))
        mask = np.array(Image.open(test_mask_path).convert('L'))
        
        print(f"  Image: {img.shape}, Mask: {mask.shape}")
        
        if not os.path.exists('big-lama'):
            print("⚠ big-lama model not found. Skipping inference.")
            return True
        
        print("Running inpainting on real image...")
        result = inpaint(img, mask, model_path='big-lama')
        
        # Save result
        Image.fromarray(result).save('test_output.jpg')
        print("✓ Saved result to test_output.jpg")
        
        return True
        
    except Exception as e:
        print(f"✗ Real image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Minimal LAMA Inpainting Implementation")
    print("=" * 60)
    
    tests = [
        ("Import test", test_import),
        ("Basic functionality test", test_basic_functionality),
        ("Real image test", test_with_real_image),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}...")
        print("-" * 60)
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
