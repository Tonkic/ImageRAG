import argparse
import glob
import os
import sys
from typing import List


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)


def load_dataset_paths(dataset_name: str) -> List[str]:
    paths = []
    if dataset_name == "aircraft":
        root = os.path.join(REPO_ROOT, "datasets", "fgvc-aircraft-2013b", "data", "images")
        list_file = os.path.join(REPO_ROOT, "datasets", "fgvc-aircraft-2013b", "data", "images_train.txt")
        if os.path.exists(list_file):
            with open(list_file, "r", encoding="utf-8") as handle:
                paths = [os.path.join(root, line.strip() + ".jpg") for line in handle if line.strip()]
    elif dataset_name == "cub":
        root = os.path.join(REPO_ROOT, "datasets", "CUB_200_2011", "images")
        split_file = os.path.join(REPO_ROOT, "datasets", "CUB_200_2011", "train_test_split.txt")
        images_file = os.path.join(REPO_ROOT, "datasets", "CUB_200_2011", "images.txt")
        if os.path.exists(split_file) and os.path.exists(images_file):
            with open(split_file, "r", encoding="utf-8") as sf:
                train_ids = {line.split()[0] for line in sf if len(line.split()) >= 2 and line.split()[1] == "1"}
            with open(images_file, "r", encoding="utf-8") as imf:
                paths = [
                    os.path.join(root, line.split()[1])
                    for line in imf
                    if len(line.split()) >= 2 and line.split()[0] in train_ids
                ]
    elif dataset_name == "imagenet":
        imagenet_list_candidates = [
            os.path.join(REPO_ROOT, "datasets", "imagenet_train_list.txt"),
            os.path.join(REPO_ROOT, "datasets", "imagenet_val_list.txt"),
        ]
        imagenet_root_candidates = [
            os.path.join(REPO_ROOT, "datasets", "ILSVRC2012_train"),
            os.path.join(REPO_ROOT, "datasets", "imagenet", "train"),
            os.path.join(REPO_ROOT, "datasets", "ILSVRC", "Data", "CLS-LOC", "train"),
            os.path.join(REPO_ROOT, "datasets", "imagenet"),
        ]
        imagenet_root = next((p for p in imagenet_root_candidates if os.path.isdir(p)), None)
        imagenet_list = next((p for p in imagenet_list_candidates if os.path.isfile(p)), None)

        if imagenet_root and imagenet_list:
            with open(imagenet_list, "r", encoding="utf-8") as handle:
                paths = [os.path.join(imagenet_root, line.strip()) for line in handle if line.strip()]
        if not paths:
            for root_candidate in imagenet_root_candidates:
                if not os.path.isdir(root_candidate):
                    continue
                paths = glob.glob(os.path.join(root_candidate, "**", "*.JPEG"), recursive=True)
                if not paths:
                    paths = glob.glob(os.path.join(root_candidate, "**", "*.jpg"), recursive=True)
                if paths:
                    break
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return paths


def load_paths_from_dir(image_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                paths.append(os.path.join(root, file))
    return sorted(paths)


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute SigLIP2 embeddings with HF/Transformers.")
    parser.add_argument("--dataset", type=str, choices=["aircraft", "cub", "imagenet"], default=None,
                        help="Known dataset shortcut. If set, image paths and output_dir default are inferred.")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Custom image directory. Use this if not using --dataset.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for siglip2_embeddings_b{offset}.pt.")
    parser.add_argument("--model_id", type=str, default="google/siglip2-so400m-patch16-naflex",
                        help="HF model id. Recommended: google/siglip2-so400m-patch16-naflex")
    parser.add_argument("--siglip2_model_id", type=str, default=None,
                        help="Alias of --model_id. If set, it overrides --model_id.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--cache_batch_size", type=int, default=1024,
                        help="Cache chunk size in filenames. Keep 1024 to match current retrieval loader.")
    parser.add_argument("--encode_batch_size", type=int, default=32,
                        help="Actual mini-batch size for model forward.")
    parser.add_argument("--max_num_patches", type=int, default=None,
                        help="Optional NaFlex max_num_patches override.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def unwrap_siglip2_features(features, modality="image"):
    import torch

    if isinstance(features, torch.Tensor):
        return features

    preferred = [f"{modality}_embeds", "pooler_output", "image_embeds", "text_embeds", "last_hidden_state"]
    for attr in preferred:
        value = getattr(features, attr, None)
        if isinstance(value, torch.Tensor):
            if value.dim() == 3:
                return value[:, 0, :]
            return value

    if isinstance(features, (tuple, list)) and features:
        first = features[0]
        if isinstance(first, torch.Tensor):
            if first.dim() == 3:
                return first[:, 0, :]
            return first

    raise TypeError(f"Unsupported SigLIP2 feature output type: {type(features)}")


def main():
    args = parse_args()
    if args.siglip2_model_id:
        args.model_id = args.siglip2_model_id

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tqdm import tqdm

    if args.dataset is None and args.image_dir is None:
        raise ValueError("Either --dataset or --image_dir must be provided.")

    if args.dataset is not None:
        image_paths = load_dataset_paths(args.dataset)
        output_dir = args.output_dir or os.path.join(REPO_ROOT, "datasets", "embeddings", args.dataset)
    else:
        if not args.image_dir or not os.path.isdir(args.image_dir):
            raise ValueError(f"Invalid --image_dir: {args.image_dir}")
        image_paths = load_paths_from_dir(args.image_dir)
        output_dir = args.output_dir or os.path.join(REPO_ROOT, "datasets", "embeddings", os.path.basename(args.image_dir.rstrip('/')))

    if not image_paths:
        raise RuntimeError("No images found to encode.")

    os.makedirs(output_dir, exist_ok=True)

    print(f"[SigLIP2] model_id={args.model_id}")
    print(f"[SigLIP2] device={args.device} dtype={args.dtype} attn={args.attn_implementation}")
    print(f"[SigLIP2] images={len(image_paths)} output_dir={output_dir}")
    print(f"[SigLIP2] cache_batch_size={args.cache_batch_size} encode_batch_size={args.encode_batch_size}")
    if args.max_num_patches is not None:
        print(f"[SigLIP2] max_num_patches={args.max_num_patches}")

    from transformers import AutoModel, AutoProcessor

    dtype = resolve_dtype(args.dtype)
    model = AutoModel.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    ).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model_id, local_files_only=args.local_files_only)

    with torch.no_grad():
        for bi in range(0, len(image_paths), args.cache_batch_size):
            cache_file = os.path.join(output_dir, f"siglip2_embeddings_b{bi}.pt")
            if os.path.exists(cache_file) and not args.overwrite:
                print(f"[Skip] {cache_file}")
                continue

            chunk_paths = image_paths[bi: bi + args.cache_batch_size]
            chunk_embs = []
            valid_paths = []

            for sj in tqdm(range(0, len(chunk_paths), args.encode_batch_size), desc=f"Chunk {bi}", leave=False):
                sub_paths = chunk_paths[sj: sj + args.encode_batch_size]
                images = []
                current_valid = []
                for path in sub_paths:
                    try:
                        images.append(Image.open(path).convert("RGB"))
                        current_valid.append(path)
                    except Exception as e:
                        print(f"[Warn] Failed to load {path}: {e}")

                if not images:
                    continue

                proc_kwargs = {"images": images, "return_tensors": "pt"}
                if args.max_num_patches is not None:
                    proc_kwargs["max_num_patches"] = args.max_num_patches
                batch = processor(**proc_kwargs)
                batch = {k: v.to(args.device) for k, v in batch.items()}

                image_features = unwrap_siglip2_features(model.get_image_features(**batch), modality="image")
                image_features = F.normalize(image_features, dim=-1).cpu()
                chunk_embs.append(image_features)
                valid_paths.extend(current_valid)

            if not chunk_embs:
                print(f"[Warn] No valid images in chunk starting at {bi}")
                continue

            embeddings = torch.cat(chunk_embs, dim=0)
            torch.save(
                {
                    "normalized_siglip2_embeddings": embeddings,
                    "paths": valid_paths,
                    "model_name": args.model_id,
                    "attn_implementation": args.attn_implementation,
                    "dtype": args.dtype,
                    "max_num_patches": args.max_num_patches,
                },
                cache_file,
            )
            print(f"[Saved] {cache_file} shape={tuple(embeddings.shape)}")

    print("✅ SigLIP2 embedding precomputation finished.")
    print("提示：如果你后续实验也要用同一 SigLIP2 模型，请在运行前设置环境变量：")
    print(f"export SIGLIP2_MODEL_ID={args.model_id}")


if __name__ == "__main__":
    main()
