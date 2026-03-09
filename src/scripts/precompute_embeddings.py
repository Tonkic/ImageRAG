import os
import glob
import argparse
import sys

# Add project root to path so src modules and Long-CLIP are correctly found
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

from src.retrieval.memory_guided_retrieval import ImageRetriever

def main():
    parser = argparse.ArgumentParser(description="Precompute visual embeddings for a given dataset offline.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Absolute or relative path to the dataset images directory (e.g., datasets/ILSVRC2012_train)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save the computed .pt embeddings (e.g., datasets/embeddings/imagenet)")
    parser.add_argument("--method", type=str, default="LongCLIP",
                        choices=["CLIP", "LongCLIP", "SigLIP", "SigLIP2", "Qwen3-VL"],
                        help="Embedding visual model to use.")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device calculation.")

    args = parser.parse_args()

    # define image directory
    if not os.path.exists(args.image_dir):
        print(f"Error: Image Directory {args.image_dir} does not exist.")
        return

    # Grab all likely images recursively (ImageNet has subfolders)
    image_paths = []
    print(f"Scanning {args.image_dir} for images (this might take a moment)...")
    for root, _, files in os.walk(args.image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f"Error: No images found in {args.image_dir}.")
        return

    print(f"Found {len(image_paths)} images.")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Precomputing {args.method} embeddings into {args.output_dir}...")
    print("This might take a while if the dataset is large. Intermediate .pt batches will be saved.")

    print("Initializing retriever to load models...")
    retriever = ImageRetriever(
        image_paths=image_paths,
        embeddings_path=args.output_dir,
        method=args.method,
        device=args.device,
        k=1,
        use_hybrid=False
    )

    print("Triggering extraction process...")
    import torch
    import torch.nn.functional as F
    from PIL import Image

    prefix_map = {
        "CLIP": "clip_embeddings",
        "LongCLIP": "longclip_embeddings",
        "SigLIP": "siglip_embeddings",
        "SigLIP2": "siglip2_embeddings"
    }
    prefix = prefix_map.get(args.method, "clip_embeddings")
    bs = 1024

    with torch.no_grad():
        for bi in range(0, len(image_paths), bs):
            cache_file = os.path.join(args.output_dir, f"{prefix}_b{bi}.pt")
            current_batch_paths = image_paths[bi : bi + bs]

            if os.path.exists(cache_file):
                print(f"Batch {bi} already exists. Skipping.")
                continue

            print(f"Processing batch {bi} to {min(bi + bs, len(image_paths))} / {len(image_paths)}...")

            if args.method in ["CLIP", "LongCLIP"]:
                images = []
                valid_paths = []
                for path in current_batch_paths:
                    try:
                        img_pil = Image.open(path).convert("RGB")
                        image = retriever.preprocess(img_pil).unsqueeze(0).to(args.device)
                        images.append(image)
                        valid_paths.append(path)
                    except Exception as e:
                        print(f"Warning: Could not read {path}: {e}")
                        continue

                if not images:
                    continue

                images = torch.cat(images, dim=0)
                image_features = retriever.model.encode_image(images)
                normalized_im_vectors = F.normalize(image_features, p=2, dim=-1)

                # Move to CPU before saving to save VRAM and disk space predictability
                torch.save({
                    f"normalized_{prefix}": normalized_im_vectors.cpu(),
                    "paths": valid_paths
                }, cache_file)
            else:
                print(f"Extraction for {args.method} is not fully implemented in this script. Missing exact preprocessing logic.")
                break

    print("✅ Precomputation finished successfully.")

if __name__ == "__main__":
    main()
