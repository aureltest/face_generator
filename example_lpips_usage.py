#!/usr/bin/env python3
"""
Example usage of Arc2Face LPIPS evaluator.
This script demonstrates how to evaluate Arc2Face-generated images
using the LPIPS perceptual similarity metric.
"""

import sys
sys.path.append('./')

from arc2face_lpips_eval import Arc2FaceLPIPSEvaluator
import os


def main():
    """
    Main example demonstrating Arc2Face LPIPS evaluation.
    """
    
    # Initialize the evaluator
    print("Initializing Arc2Face LPIPS Evaluator...")
    evaluator = Arc2FaceLPIPSEvaluator(
        device="cuda" if torch.cuda.is_available() else "cpu",
        lpips_net="alex",  # 'alex' is best for evaluation
        image_size=256,    # Standard size for LPIPS
        arc2face_model_path="models",
        base_model="stable-diffusion-v1-5/stable-diffusion-v1-5"
    )
    
    # Example 1: Evaluate a single image
    print("\n=== Example 1: Single Image Evaluation ===")
    
    # Use one of the example images
    test_image = "assets/examples/freeman.jpg"
    
    if os.path.exists(test_image):
        # Evaluate with different generation parameters
        results = evaluator.evaluate_single_image(
            test_image,
            num_generated=4,
            generation_params={
                "num_inference_steps": 25,
                "guidance_scale": 3.0,
                "seed": 42  # For reproducibility
            }
        )
        
        if "error" not in results:
            print(f"Original image: {results['original_image']}")
            print(f"Generated {results['num_generated']} images")
            print(f"LPIPS scores: {[f'{score:.4f}' for score in results['lpips_scores']]}")
            print(f"Mean LPIPS: {results['mean_lpips']:.4f}")
            print(f"Std LPIPS: {results['std_lpips']:.4f}")
            print(f"Range: [{results['min_lpips']:.4f}, {results['max_lpips']:.4f}]")
            
            # Interpret results
            if results['mean_lpips'] < 0.5:
                print("✅ Excellent perceptual similarity!")
            elif results['mean_lpips'] < 0.7:
                print("✅ Good perceptual similarity")
            else:
                print("⚠️ Moderate perceptual similarity")
    
    # Example 2: Batch evaluation on multiple images
    print("\n=== Example 2: Batch Evaluation ===")
    
    # List of images to evaluate
    image_list = [
        "assets/examples/freeman.jpg",
        "assets/examples/lily.png",
        "assets/examples/joacquin.png",
        "assets/examples/jackie.png"
    ]
    
    # Filter existing images
    existing_images = [img for img in image_list if os.path.exists(img)]
    
    if len(existing_images) >= 2:
        print(f"Evaluating {len(existing_images)} images...")
        
        # Run batch evaluation
        batch_results = evaluator.evaluate_dataset(
            existing_images,
            num_generated_per_image=3,
            generation_params={
                "num_inference_steps": 25,
                "guidance_scale": 3.0,
                "seed": 123
            },
            save_results=True,
            output_dir="arc2face_lpips_results"
        )
        
        # Print summary statistics
        print("\n📊 Batch Evaluation Summary:")
        print(f"Images evaluated: {batch_results['num_images_evaluated']}")
        print(f"Total generations: {batch_results['total_generations']}")
        print(f"Overall mean LPIPS: {batch_results['overall_mean_lpips']:.4f}")
        print(f"Overall std LPIPS: {batch_results['overall_std_lpips']:.4f}")
        
        print("\n📈 Distribution:")
        for percentile, value in batch_results['percentiles'].items():
            print(f"  {percentile} percentile: {value:.4f}")
        
        print("\n✅ Results saved to 'arc2face_lpips_results/' directory")
        print("   - evaluation_results.json: Detailed metrics")
        print("   - lpips_distribution.png: Visual analysis")
        print("   - {image_name}/: Generated images for each input")
    
    # Example 3: Custom evaluation workflow
    print("\n=== Example 3: Custom Evaluation Workflow ===")
    
    if os.path.exists("assets/examples/freeman.jpg"):
        # Extract face embedding manually
        face_emb = evaluator.extract_face_embedding("assets/examples/freeman.jpg")
        
        if face_emb is not None:
            # Generate images with different parameters
            print("Testing different generation parameters...")
            
            # Test 1: Different number of steps
            for steps in [10, 25, 50]:
                images = evaluator.generate_images(
                    face_emb,
                    num_images=1,
                    num_inference_steps=steps,
                    guidance_scale=3.0,
                    seed=42
                )
                
                lpips_score = evaluator.compute_lpips(
                    "assets/examples/freeman.jpg",
                    images[0]
                )
                print(f"  Steps={steps}: LPIPS={lpips_score:.4f}")
            
            print("\n")
            
            # Test 2: Different guidance scales
            for guidance in [1.0, 3.0, 5.0, 7.0]:
                images = evaluator.generate_images(
                    face_emb,
                    num_images=1,
                    num_inference_steps=25,
                    guidance_scale=guidance,
                    seed=42
                )
                
                lpips_score = evaluator.compute_lpips(
                    "assets/examples/freeman.jpg",
                    images[0]
                )
                print(f"  Guidance={guidance}: LPIPS={lpips_score:.4f}")
    
    print("\n=== Evaluation Complete! ===")
    print("\nInterpretation Guide:")
    print("LPIPS < 0.1: Excellent similarity (nearly identical)")
    print("LPIPS 0.1-0.3: Very good similarity")
    print("LPIPS 0.3-0.5: Good similarity")
    print("LPIPS 0.5-0.7: Moderate similarity")
    print("LPIPS > 0.7: Low similarity")


if __name__ == "__main__":
    import torch
    main()