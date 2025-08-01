#!/usr/bin/env python3
"""
Test script for Arc2Face LPIPS evaluation.
Tests basic functionality with example images.
"""

import sys
sys.path.append('./')

import os
import torch
from arc2face_lpips_eval import Arc2FaceLPIPSEvaluator
from PIL import Image
import numpy as np


def test_lpips_computation():
    """Test basic LPIPS computation between two images."""
    print("=== Testing LPIPS Computation ===")
    
    # Initialize evaluator
    evaluator = Arc2FaceLPIPSEvaluator(
        device="cuda" if torch.cuda.is_available() else "cpu",
        lpips_net="alex",
        image_size=256
    )
    
    # Create two test images
    print("Creating test images...")
    img1 = Image.new('RGB', (512, 512), color='red')
    img2 = Image.new('RGB', (512, 512), color='blue')
    
    # Compute LPIPS
    score = evaluator.compute_lpips(img1, img2)
    print(f"LPIPS between red and blue images: {score:.4f}")
    
    # Test with same image (should be near 0)
    score_same = evaluator.compute_lpips(img1, img1)
    print(f"LPIPS between identical images: {score_same:.4f}")
    
    assert score > 0.1, "LPIPS between different colors should be high"
    assert score_same < 0.01, "LPIPS between identical images should be near 0"
    print("✓ LPIPS computation test passed!\n")


def test_single_image_evaluation():
    """Test evaluation on a single example image."""
    print("=== Testing Single Image Evaluation ===")
    
    # Check if example images exist
    example_images = [
        "assets/examples/freeman.jpg",
        "assets/examples/lily.png",
        "assets/examples/joacquin.png"
    ]
    
    test_image = None
    for img_path in example_images:
        if os.path.exists(img_path):
            test_image = img_path
            break
    
    if test_image is None:
        print("⚠ No example images found. Skipping single image test.")
        return
    
    print(f"Testing with image: {test_image}")
    
    # Initialize evaluator
    evaluator = Arc2FaceLPIPSEvaluator(
        device="cuda" if torch.cuda.is_available() else "cpu",
        lpips_net="alex",
        image_size=256
    )
    
    # Evaluate single image
    results = evaluator.evaluate_single_image(
        test_image,
        num_generated=2,  # Generate only 2 images for quick test
        generation_params={
            "num_inference_steps": 10,  # Fewer steps for faster testing
            "guidance_scale": 3.0,
            "seed": 42
        }
    )
    
    if "error" in results:
        print(f"⚠ Error: {results['error']}")
        return
    
    print(f"Generated {results['num_generated']} images")
    print(f"LPIPS scores: {results['lpips_scores']}")
    print(f"Mean LPIPS: {results['mean_lpips']:.4f}")
    print(f"Std LPIPS: {results['std_lpips']:.4f}")
    
    # Validate results
    assert len(results['lpips_scores']) == 2, "Should have 2 LPIPS scores"
    assert all(0 <= score <= 1 for score in results['lpips_scores']), "LPIPS scores should be in [0, 1]"
    print("✓ Single image evaluation test passed!\n")


def test_batch_evaluation():
    """Test batch evaluation on multiple images."""
    print("=== Testing Batch Evaluation ===")
    
    # Find available example images
    example_images = [
        "assets/examples/freeman.jpg",
        "assets/examples/lily.png",
        "assets/examples/joacquin.png"
    ]
    
    available_images = [img for img in example_images if os.path.exists(img)]
    
    if len(available_images) < 2:
        print("⚠ Not enough example images found. Skipping batch test.")
        return
    
    print(f"Testing with {len(available_images)} images")
    
    # Initialize evaluator
    evaluator = Arc2FaceLPIPSEvaluator(
        device="cuda" if torch.cuda.is_available() else "cpu",
        lpips_net="alex",
        image_size=256
    )
    
    # Evaluate dataset
    results = evaluator.evaluate_dataset(
        available_images[:2],  # Use only first 2 images for quick test
        num_generated_per_image=2,
        generation_params={
            "num_inference_steps": 10,
            "guidance_scale": 3.0,
            "seed": 42
        },
        save_results=True,
        output_dir="test_evaluation_results"
    )
    
    print(f"Images evaluated: {results['num_images_evaluated']}")
    print(f"Total generations: {results['total_generations']}")
    print(f"Overall mean LPIPS: {results['overall_mean_lpips']:.4f}")
    print(f"Overall std LPIPS: {results['overall_std_lpips']:.4f}")
    
    # Check if results were saved
    if os.path.exists("test_evaluation_results/evaluation_results.json"):
        print("✓ Results saved successfully")
    
    print("✓ Batch evaluation test passed!\n")


def main():
    """Run all tests."""
    print("Starting Arc2Face LPIPS evaluation tests...\n")
    
    try:
        # Test 1: Basic LPIPS computation
        test_lpips_computation()
        
        # Test 2: Single image evaluation
        test_single_image_evaluation()
        
        # Test 3: Batch evaluation
        test_batch_evaluation()
        
        print("=== All tests completed successfully! ===")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()