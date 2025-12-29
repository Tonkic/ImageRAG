
import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import numpy as np

MODEL_NAME = "BAAI/BGE-VL-large"

print(f"Loading {MODEL_NAME}...")
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()

print("Model loaded.")

# Check Vision Model Structure
if hasattr(model, "vision_model"):
    vision_model = model.vision_model
    print("Found model.vision_model")
elif hasattr(model, "visual_model"):
    vision_model = model.visual_model
    print("Found model.visual_model")
else:
    print("Could not find vision model.")
    exit()

print(f"Vision Config: {vision_model.config}")

if hasattr(vision_model, "embeddings"):
    embeddings = vision_model.embeddings
    print(f"Embeddings type: {type(embeddings)}")
    print(f"Embeddings attributes: {dir(embeddings)}")

    if hasattr(embeddings, "position_ids"):
        print(f"Found position_ids in embeddings. Shape: {embeddings.position_ids.shape}")
        print(f"Max pos id: {embeddings.position_ids.max()}")
        print(f"Min pos id: {embeddings.position_ids.min()}")

    if hasattr(embeddings, "position_embedding"):
        if hasattr(embeddings.position_embedding, "weight"):
            print(f"Position Embedding Weight Shape: {embeddings.position_embedding.weight.shape}")

# Check Processor
print("Setting processor...")
model.set_processor(MODEL_NAME)
processor = model.processor
print(f"Processor: {processor}")

# Dummy Image
image = Image.new('RGB', (1000, 1000), color='red')
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs['pixel_values']
print(f"Pixel Values Shape: {pixel_values.shape}")

# Try Encode Image on CPU
print("Attempting encode_image on CPU...")
try:
    out = model.encode_image(pixel_values)
    print("Encode Image Success!")
    print(f"Output shape: {out.shape}")
except Exception as e:
    print(f"Encode Image Failed: {e}")
    import traceback
    traceback.print_exc()
