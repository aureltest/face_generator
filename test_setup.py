import torch
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("\nPyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU device:", torch.cuda.get_device_name(0))
    print("GPU memory:", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("\nTesting imports...")
try:
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
    print("✓ diffusers imported successfully")
except ImportError as e:
    print("✗ diffusers import failed:", e)

try:
    from transformers import CLIPTextModel
    print("✓ transformers imported successfully")
except ImportError as e:
    print("✗ transformers import failed:", e)

try:
    from insightface.app import FaceAnalysis
    print("✓ insightface imported successfully")
except ImportError as e:
    print("✗ insightface import failed:", e)

try:
    from arc2face import CLIPTextModelWrapper, project_face_embs
    print("✓ arc2face imported successfully")
except ImportError as e:
    print("✗ arc2face import failed:", e)
    print("  (This is expected if arc2face.py is not in the current directory)")

print("\nModel files check:")
import os
models_dir = "./models"
required_files = [
    "arc2face/config.json",
    "arc2face/diffusion_pytorch_model.safetensors",
    "encoder/config.json",
    "encoder/pytorch_model.bin",
    "antelopev2/arcface.onnx"
]

for file in required_files:
    path = os.path.join(models_dir, file)
    if os.path.exists(path):
        print(f"✓ {file} exists")
    else:
        print(f"✗ {file} missing")

print("\n⚠️  Note: You still need to manually download the antelopev2 package from:")
print("https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view")
print("Extract it and place the contents in ./models/antelopev2/")
print("Then delete glintr100.onnx from the antelopev2 folder.")