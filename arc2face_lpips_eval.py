import sys
sys.path.append('./')

import torch
import numpy as np
from PIL import Image
import lpips
from torchvision import transforms
from typing import List, Dict, Union, Optional, Tuple
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)
from arc2face import CLIPTextModelWrapper, project_face_embs
from insightface.app import FaceAnalysis


class Arc2FaceLPIPSEvaluator:
    """
    Evaluator class for assessing Arc2Face generation quality using LPIPS metric.
    """
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lpips_net: str = "alex",
        image_size: int = 256,
        arc2face_model_path: str = "models",
        base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    ):
        """
        Initialize the evaluator with Arc2Face and LPIPS models.
        
        Args:
            device: Device to run models on
            lpips_net: LPIPS network type ('alex', 'vgg', 'squeeze')
            image_size: Size to resize images for LPIPS evaluation
            arc2face_model_path: Path to Arc2Face model files
            base_model: Base Stable Diffusion model identifier
        """
        self.device = device
        self.image_size = image_size
        
        # Initialize LPIPS
        print(f"Initializing LPIPS with {lpips_net} network...")
        self.lpips_fn = lpips.LPIPS(net=lpips_net).to(device)
        
        # Initialize Arc2Face pipeline
        print("Loading Arc2Face model...")
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        
        self.encoder = CLIPTextModelWrapper.from_pretrained(
            arc2face_model_path, subfolder="encoder", torch_dtype=self.dtype
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            arc2face_model_path, subfolder="arc2face", torch_dtype=self.dtype
        )
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            base_model,
            text_encoder=self.encoder,
            unet=self.unet,
            torch_dtype=self.dtype,
            safety_checker=None
        )
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        self.pipeline = self.pipeline.to(device)
        
        # Initialize face analysis
        print("Loading face analysis model...")
        self.face_app = FaceAnalysis(
            name='antelopev2',
            root='./',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Image preprocessing for LPIPS
        self.preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
    
    def extract_face_embedding(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Extract ArcFace embedding from an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Face embedding tensor or None if no face detected
        """
        img = np.array(Image.open(image_path))[:, :, ::-1]  # RGB to BGR
        faces = self.face_app.get(img)
        
        if len(faces) == 0:
            print(f"No face detected in {image_path}")
            return None
        
        # Select largest face
        face = sorted(faces, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * 
                                          (x['bbox'][3] - x['bbox'][1]))[-1]
        
        id_emb = torch.tensor(face['embedding'], dtype=self.dtype)[None].to(self.device)
        id_emb = id_emb / torch.norm(id_emb, dim=1, keepdim=True)  # Normalize
        id_emb = project_face_embs(self.pipeline, id_emb)
        
        return id_emb
    
    def generate_images(
        self,
        face_embedding: torch.Tensor,
        num_images: int = 4,
        num_inference_steps: int = 25,
        guidance_scale: float = 3.0,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate images using Arc2Face.
        
        Args:
            face_embedding: Projected face embedding
            num_images: Number of images to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            seed: Random seed for reproducibility
            
        Returns:
            List of generated PIL images
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        images = self.pipeline(
            prompt_embeds=face_embedding,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator
        ).images
        
        return images
    
    def compute_lpips(
        self,
        img1: Union[str, Image.Image],
        img2: Union[str, Image.Image]
    ) -> float:
        """
        Compute LPIPS distance between two images.
        
        Args:
            img1: First image (path or PIL Image)
            img2: Second image (path or PIL Image)
            
        Returns:
            LPIPS distance (lower is more similar)
        """
        # Load images if paths provided
        if isinstance(img1, str):
            img1 = Image.open(img1).convert('RGB')
        if isinstance(img2, str):
            img2 = Image.open(img2).convert('RGB')
        
        # Preprocess images
        img1_tensor = self.preprocess(img1).unsqueeze(0).to(self.device)
        img2_tensor = self.preprocess(img2).unsqueeze(0).to(self.device)
        
        # Compute LPIPS
        with torch.no_grad():
            distance = self.lpips_fn(img1_tensor, img2_tensor)
        
        return distance.item()
    
    def evaluate_single_image(
        self,
        original_image_path: str,
        num_generated: int = 4,
        generation_params: Optional[Dict] = None
    ) -> Dict:
        """
        Evaluate Arc2Face generation for a single original image.
        
        Args:
            original_image_path: Path to the original image
            num_generated: Number of images to generate
            generation_params: Optional generation parameters
            
        Returns:
            Dictionary with evaluation results
        """
        if generation_params is None:
            generation_params = {}
        
        # Extract face embedding
        face_embedding = self.extract_face_embedding(original_image_path)
        if face_embedding is None:
            return {"error": "No face detected in original image"}
        
        # Generate images
        generated_images = self.generate_images(
            face_embedding,
            num_images=num_generated,
            **generation_params
        )
        
        # Compute LPIPS scores
        original_img = Image.open(original_image_path).convert('RGB')
        lpips_scores = []
        
        for gen_img in generated_images:
            score = self.compute_lpips(original_img, gen_img)
            lpips_scores.append(score)
        
        # Compute statistics
        results = {
            "original_image": original_image_path,
            "num_generated": num_generated,
            "lpips_scores": lpips_scores,
            "mean_lpips": np.mean(lpips_scores),
            "std_lpips": np.std(lpips_scores),
            "min_lpips": np.min(lpips_scores),
            "max_lpips": np.max(lpips_scores),
            "generated_images": generated_images
        }
        
        return results
    
    def evaluate_dataset(
        self,
        image_paths: List[str],
        num_generated_per_image: int = 4,
        generation_params: Optional[Dict] = None,
        save_results: bool = True,
        output_dir: str = "evaluation_results"
    ) -> Dict:
        """
        Evaluate Arc2Face on a dataset of images.
        
        Args:
            image_paths: List of paths to original images
            num_generated_per_image: Number of generations per image
            generation_params: Optional generation parameters
            save_results: Whether to save results to disk
            output_dir: Directory to save results
            
        Returns:
            Dictionary with aggregated evaluation results
        """
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        all_results = []
        all_lpips_scores = []
        
        for img_path in tqdm(image_paths, desc="Evaluating images"):
            result = self.evaluate_single_image(
                img_path,
                num_generated_per_image,
                generation_params
            )
            
            if "error" not in result:
                all_results.append(result)
                all_lpips_scores.extend(result["lpips_scores"])
                
                if save_results:
                    # Save generated images
                    img_name = os.path.splitext(os.path.basename(img_path))[0]
                    img_dir = os.path.join(output_dir, img_name)
                    os.makedirs(img_dir, exist_ok=True)
                    
                    for i, gen_img in enumerate(result["generated_images"]):
                        gen_img.save(os.path.join(img_dir, f"generated_{i}.png"))
        
        # Aggregate results
        aggregated_results = {
            "num_images_evaluated": len(all_results),
            "total_generations": len(all_lpips_scores),
            "overall_mean_lpips": np.mean(all_lpips_scores),
            "overall_std_lpips": np.std(all_lpips_scores),
            "overall_min_lpips": np.min(all_lpips_scores),
            "overall_max_lpips": np.max(all_lpips_scores),
            "percentiles": {
                "25th": np.percentile(all_lpips_scores, 25),
                "50th": np.percentile(all_lpips_scores, 50),
                "75th": np.percentile(all_lpips_scores, 75),
                "90th": np.percentile(all_lpips_scores, 90),
                "95th": np.percentile(all_lpips_scores, 95)
            },
            "individual_results": all_results
        }
        
        if save_results:
            # Save JSON results
            json_path = os.path.join(output_dir, "evaluation_results.json")
            with open(json_path, 'w') as f:
                # Remove PIL images from results before saving
                save_results = aggregated_results.copy()
                for result in save_results["individual_results"]:
                    result.pop("generated_images", None)
                json.dump(save_results, f, indent=2)
            
            # Generate and save visualization
            self.visualize_results(all_lpips_scores, output_dir)
        
        return aggregated_results
    
    def visualize_results(self, lpips_scores: List[float], output_dir: str):
        """
        Create and save visualization of LPIPS scores.
        
        Args:
            lpips_scores: List of LPIPS scores
            output_dir: Directory to save plots
        """
        plt.figure(figsize=(12, 4))
        
        # Histogram
        plt.subplot(1, 3, 1)
        plt.hist(lpips_scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('LPIPS Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of LPIPS Scores')
        plt.axvline(np.mean(lpips_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(lpips_scores):.3f}')
        plt.legend()
        
        # Box plot
        plt.subplot(1, 3, 2)
        plt.boxplot(lpips_scores)
        plt.ylabel('LPIPS Score')
        plt.title('LPIPS Score Box Plot')
        
        # CDF
        plt.subplot(1, 3, 3)
        sorted_scores = np.sort(lpips_scores)
        p = np.arange(len(sorted_scores)) / float(len(sorted_scores) - 1)
        plt.plot(sorted_scores, p)
        plt.xlabel('LPIPS Score')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution Function')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lpips_distribution.png'), dpi=300)
        plt.close()


def main():
    """
    Example usage of the Arc2Face LPIPS evaluator.
    """
    # Initialize evaluator
    evaluator = Arc2FaceLPIPSEvaluator(
        device="cuda" if torch.cuda.is_available() else "cpu",
        lpips_net="alex",
        image_size=256
    )
    
    # Example: Evaluate single image
    example_images = [
        "assets/examples/freeman.jpg",
        "assets/examples/lily.png",
        "assets/examples/joacquin.png"
    ]
    
    # Filter existing images
    existing_images = [img for img in example_images if os.path.exists(img)]
    
    if existing_images:
        print(f"\nEvaluating {len(existing_images)} example images...")
        
        # Evaluate dataset
        results = evaluator.evaluate_dataset(
            existing_images,
            num_generated_per_image=4,
            generation_params={
                "num_inference_steps": 25,
                "guidance_scale": 3.0,
                "seed": 42
            },
            save_results=True,
            output_dir="lpips_evaluation_results"
        )
        
        # Print summary
        print("\n=== Evaluation Summary ===")
        print(f"Images evaluated: {results['num_images_evaluated']}")
        print(f"Total generations: {results['total_generations']}")
        print(f"Mean LPIPS: {results['overall_mean_lpips']:.4f}")
        print(f"Std LPIPS: {results['overall_std_lpips']:.4f}")
        print(f"Min LPIPS: {results['overall_min_lpips']:.4f}")
        print(f"Max LPIPS: {results['overall_max_lpips']:.4f}")
        print("\nPercentiles:")
        for percentile, value in results['percentiles'].items():
            print(f"  {percentile}: {value:.4f}")
    else:
        print("No example images found. Please provide image paths.")


if __name__ == "__main__":
    main()