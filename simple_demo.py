import torch
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, '.')

print("Setting up Arc2Face...")

# Import Arc2Face components
from arc2face import CLIPTextModelWrapper, project_face_embs
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from insightface.app import FaceAnalysis

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Arc2Face is built upon SD1.5
base_model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'

print("\nLoading models...")

# Load encoder
encoder = CLIPTextModelWrapper.from_pretrained(
    'models', subfolder="encoder", torch_dtype=torch.float16
)

# Load UNet
unet = UNet2DConditionModel.from_pretrained(
    'models', subfolder="arc2face", torch_dtype=torch.float16
)

# Create pipeline
pipeline = StableDiffusionPipeline.from_pretrained(
    base_model,
    text_encoder=encoder,
    unet=unet,
    torch_dtype=torch.float16,
    safety_checker=None
)

# Set scheduler
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to(device)

print("\nPipeline loaded successfully!")

# Initialize face analysis
print("\nInitializing face analysis...")
app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load example image
img_path = 'assets/examples/joacquin.png'
print(f"\nLoading example image: {img_path}")
img = np.array(Image.open(img_path))[:,:,::-1]

# Extract face embedding
faces = app.get(img)
if len(faces) == 0:
    print("No face detected!")
    sys.exit(1)

# Select largest face if multiple detected
face = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
id_emb = torch.tensor(face['embedding'], dtype=torch.float16)[None].to(device)
id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True)   # normalize embedding
id_emb = project_face_embs(pipeline, id_emb)    # pass through the encoder

print("\nGenerating images...")
num_images = 4
images = pipeline(prompt_embeds=id_emb, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=num_images).images

# Save images
for i, img in enumerate(images):
    img.save(f'output_{i}.png')
    print(f"Saved output_{i}.png")

print("\nDone! Generated images saved as output_0.png to output_3.png")