from huggingface_hub import hf_hub_download
import os

# Create models directory
os.makedirs("./models/arc2face", exist_ok=True)
os.makedirs("./models/encoder", exist_ok=True)
os.makedirs("./models/antelopev2", exist_ok=True)

print("Downloading Arc2Face models...")

# Download Arc2Face model
print("1/6: Downloading arc2face/config.json...")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arc2face/config.json", local_dir="./models")

print("2/6: Downloading arc2face/diffusion_pytorch_model.safetensors...")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arc2face/diffusion_pytorch_model.safetensors", local_dir="./models")

# Download encoder
print("3/6: Downloading encoder/config.json...")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="encoder/config.json", local_dir="./models")

print("4/6: Downloading encoder/pytorch_model.bin...")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="encoder/pytorch_model.bin", local_dir="./models")

# Download ArcFace recognition model
print("5/6: Downloading arcface.onnx...")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arcface.onnx", local_dir="./models/antelopev2")

print("\nAll Arc2Face models downloaded successfully!")
print("\nNote: You still need to manually download the antelopev2 package from:")
print("https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view")
print("Extract it and place the contents in ./models/antelopev2/")
print("Then delete glintr100.onnx from the antelopev2 folder.")