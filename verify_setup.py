#!/usr/bin/env python3
import sys
import torch

print("=== Arc2Face Setup Verification ===\n")

# Check Python version
print(f"Python version: {sys.version}")

# Check PyTorch and CUDA
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("\n=== Testing imports ===")

# Test core imports
try:
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
    print("✓ diffusers imported successfully")
except ImportError as e:
    print(f"✗ diffusers import failed: {e}")

try:
    from transformers import CLIPTextModel
    print("✓ transformers imported successfully")
except ImportError as e:
    print(f"✗ transformers import failed: {e}")

try:
    from insightface.app import FaceAnalysis
    print("✓ insightface imported successfully")
except ImportError as e:
    print(f"✗ insightface import failed: {e}")

try:
    import numpy as np
    print(f"✓ numpy imported successfully (version: {np.__version__})")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")

try:
    from PIL import Image
    print("✓ PIL imported successfully")
except ImportError as e:
    print(f"✗ PIL import failed: {e}")

# Try importing Arc2Face
sys.path.insert(0, '.')
try:
    from arc2face import CLIPTextModelWrapper, project_face_embs
    print("✓ arc2face imported successfully")
except ImportError as e:
    print(f"✗ arc2face import failed: {e}")

print("\n=== Checking model files ===")
import os

models_dir = "./models"
required_files = [
    "arc2face/config.json",
    "arc2face/diffusion_pytorch_model.safetensors",
    "encoder/config.json",
    "encoder/pytorch_model.bin",
    "antelopev2/arcface.onnx",
    "antelopev2/1k3d68.onnx",
    "antelopev2/2d106det.onnx",
    "antelopev2/genderage.onnx",
    "antelopev2/scrfd_10g_bnkps.onnx"
]

all_files_present = True
for file in required_files:
    path = os.path.join(models_dir, file)
    if os.path.exists(path):
        print(f"✓ {file}")
    else:
        print(f"✗ {file} - MISSING!")
        all_files_present = False

if all_files_present:
    print("\n✅ All required model files are present!")
else:
    print("\n❌ Some model files are missing!")

print("\n=== Testing basic functionality ===")
try:
    # Test face analysis initialization
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("✓ FaceAnalysis initialized successfully")
except Exception as e:
    print(f"✗ FaceAnalysis initialization failed: {e}")

print("\n=== Setup verification complete ===")
print("\nTo run Arc2Face:")
print("1. For the demo script: python3 simple_demo.py")
print("2. For the Gradio interface: python3 gradio_demo/app.py")