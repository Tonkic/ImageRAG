'''
python precompute_latents.py --omnigen2_path ./OmniGen2
'''
import os
import sys
import torch
import argparse
from tqdm import tqdm
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Precompute Visual Latents for OmniGen RAG")
    parser.add_argument("--omnigen2_path", type=str, default="./OmniGen2")
    parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2")
    parser.add_argument("--dataset_root", type=str, default="datasets/fgvc-aircraft-2013b/data/images")
    parser.add_argument("--output_dir", type=str, default="datasets/latents/aircraft")
    parser.add_argument("--device_id", type=str, default="0")
    args = parser.parse_args()

    # 1. Setup Environment
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(script_dir, args.omnigen2_path)))

    try:
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
    except ImportError:
        print("Error: Could not import OmniGen2Pipeline. Check omnigen2_path.")
        sys.exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading OmniGen Pipeline on {device}...")
    pipe = OmniGen2Pipeline.from_pretrained(args.omnigen2_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    pipe.to(device)
    pipe.vae.eval()

    # 2. Process Images
    os.makedirs(args.output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(args.dataset_root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(image_files)} images in {args.dataset_root}")

    skipped = 0
    with torch.no_grad():
        for img_name in tqdm(image_files):
            out_path = os.path.join(args.output_dir, os.path.splitext(img_name)[0] + ".pt")
            if os.path.exists(out_path):
                continue

            img_path = os.path.join(args.dataset_root, img_name)
            try:
                # 1. Load Image
                img = Image.open(img_path).convert("RGB")

                # 2. Preprocess (Reuse Pipeline Logic)
                # Pipeline defaults: max_pixels=1024*1024, max_side_length=1024
                # We replicate what prepare_image does for a single image segment.
                # prepare_image loops over list of images.

                # OmniGen2ImageProcessor.preprocess returns a Tensor [1, C, H, W]
                # It handles resizing internally.
                proc_img = pipe.image_processor.preprocess(img, max_pixels=1024*1024, max_side_length=1024)

                # 3. Encode (Reuse Pipeline Logic)
                # encode_vae handles shift/scale factors automatically
                latents = pipe.encode_vae(proc_img.to(device=device, dtype=pipe.vae.dtype))
                # Returns [1, C, H, W]

                # 4. Save (Move to CPU)
                # We save the tensor directly. Squeeze batch dim [1, C, H, W] -> [C, H, W]
                torch.save(latents.squeeze(0).cpu(), out_path)

            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                skipped += 1

    print(f"Done. Skipped {skipped} errors.")
    print(f"Latents saved to {args.output_dir}")

if __name__ == "__main__":
    main()
