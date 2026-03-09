import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import argparse
import os
import time

def main():
    parser = argparse.ArgumentParser(description="Test FLUX.1 Kontext [dev]")
    parser.add_argument("--image_path", type=str,
                        default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png",
                        help="Path or URL to input image")
    parser.add_argument("--prompt", type=str, default="Add a hat to the cat", help="Edit instruction")
    parser.add_argument("--output_path", type=str, default="flux_kontext_test.png", help="Output image path")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="Guidance scale")
    parser.add_argument("--model_path", type=str, default="/home/tingyu/imageRAG/FLUX.1-Kontext-dev", help="Path to local FLUX model or HF repo ID")

    args = parser.parse_args()

    # 1. Load pipeline
    print(f"[{time.strftime('%H:%M:%S')}] Loading FLUX.1 Kontext Pipeline from {args.model_path}...")
    pipe = FluxKontextPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16
    )

    # Enable memory saving features if on CUDA
    if args.device.startswith("cuda"):
        print(f"[{time.strftime('%H:%M:%S')}] Enabling model CPU offload to save VRAM...")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)

    print(f"[{time.strftime('%H:%M:%S')}] Model loaded to {args.device}.")

    # 2. Load input image
    print(f"[{time.strftime('%H:%M:%S')}] Loading image from {args.image_path}...")
    input_image = load_image(args.image_path)

    # 3. Generate
    print(f"[{time.strftime('%H:%M:%S')}] Running generation...")
    print(f"  Prompt: '{args.prompt}'")

    start_time = time.time()

    image = pipe(
        image=input_image,
        prompt=args.prompt,
        guidance_scale=args.guidance_scale
    ).images[0]

    end_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Generation complete in {end_time - start_time:.2f} seconds.")

    # 4. Save
    image.save(args.output_path)
    print(f"[{time.strftime('%H:%M:%S')}] Saved result to {args.output_path}")

if __name__ == "__main__":
    main()
