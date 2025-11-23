import argparse
import sys
import os
import torch
import clip
from open_clip import create_model_from_pretrained, get_tokenizer
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm

# --------------------------------------------------
# --- 数据库配置中心 (与 smart_rag_dispatcher.py 匹配) ---
# --------------------------------------------------
DATASET_CONFIGS = {
    "aircraft": {
        "train_list": "datasets/fgvc-aircraft-2013b/data/images_train.txt",
        "image_root": "datasets/fgvc-aircraft-2013b/data/images",
    },
    "cub": {
        "train_list": "datasets/CUB_200_2011/images.txt",
        "image_root": "datasets/CUB_200_2011/images",
    },
    "imagenet": {
        "train_list": "datasets/imagenet_train_list.txt",
        "image_root": "datasets/ILSVRC2012_train",
    }
}
# --------------------------------------------------

def get_image_paths(dataset_name, config):
    """ 加载图像路径列表 """
    retrieval_image_paths = []
    print(f"G-loading RAG database from {config['train_list']}...")
    try:
        with open(config['train_list'], 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line: continue
                if dataset_name == 'aircraft':
                    image_path = os.path.join(config['image_root'], f"{line}.jpg")
                elif dataset_name == 'cub':
                    image_filename = line.split(' ')[-1]
                    image_path = os.path.join(config['image_root'], image_filename)
                elif dataset_name == 'imagenet':
                    image_path = os.path.join(config['image_root'], line)
                else:
                    image_path = os.path.join(config['image_root'], line)
                if os.path.exists(image_path):
                    retrieval_image_paths.append(image_path)
        print(f"Found {len(retrieval_image_paths)} images for retrieval.")
    except FileNotFoundError:
        print(f"Error: Could not find {config['train_list']}.")
        sys.exit(1)
    return retrieval_image_paths

def process_batch(model, preprocess, image_paths, device, model_type='clip'):
    """ 处理单个批次的图像 """
    images = []
    final_paths = []
    for path in image_paths:
        try:
            image = preprocess(Image.open(path)).unsqueeze(0).to(device)
            images.append(image)
            final_paths.append(path)
        except Exception as e:
            print(f"Warning: Couldn't read {path}, skipping. Error: {e}")
            continue

    if not images:
        return None, None

    images = torch.stack(images).squeeze(1).to(device)

    with torch.no_grad():
        if model_type == 'clip':
            image_features = model.encode_image(images)
            normalized_im_vectors = F.normalize(image_features, p=2, dim=1)
        elif model_type == 'siglip':
            image_features = model.encode_image(images)
            normalized_im_vectors = F.normalize(image_features, dim=-1)

    return normalized_im_vectors, final_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Cache Generator")
    parser.add_argument("--dataset_name", type=str, required=True, choices=['aircraft', 'cub', 'imagenet'])
    parser.add_argument("--device_id", type=int, required=True, help="GPU device ID (e.g., 0)")
    parser.add_argument("--model_type", type=str, default="clip", choices=['clip', 'siglip'], help="Model type to generate embeddings for")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for processing images (e.g., 2048 for CLIP, 64 for SigLIP)")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    device = "cuda"

    # 1. 加载配置
    config = DATASET_CONFIGS[args.dataset_name]
    all_image_paths = get_image_paths(args.dataset_name, config)

    # 2. 创建输出目录
    embeddings_path = f"datasets/embeddings/{args.dataset_name}"
    os.makedirs(embeddings_path, exist_ok=True)
    print(f"Embeddings will be saved to: {embeddings_path}")

    # 3. 加载模型
    if args.model_type == 'clip':
        model, preprocess = clip.load("ViT-B/32", device=device)
        embedding_file_prefix = "clip_embeddings"
        embedding_key = "normalized_clip_embeddings"
    elif args.model_type == 'siglip':
        model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384', cache_dir=None, device=device)
        embedding_file_prefix = "siglip_embeddings"
        embedding_key = "normalized_siglip_embeddings"

    # 4. 迭代和处理
    bs = args.batch_size

    for bi in tqdm(range(0, len(all_image_paths), bs), desc=f"Generating {args.model_type} embeddings"):

        output_filename = os.path.join(embeddings_path, f"{embedding_file_prefix}_b{bi}.pt")

        if os.path.exists(output_filename):
            print(f"Skipping batch {bi}, file already exists.")
            continue

        batch_paths = all_image_paths[bi : bi + bs]
        if not batch_paths:
            continue

        print(f"\nProcessing batch {bi} / {len(all_image_paths)}")

        normalized_im_vectors, final_bi_paths = process_batch(model, preprocess, batch_paths, device, args.model_type)

        if normalized_im_vectors is None:
            print(f"Batch {bi} had no valid images.")
            continue

        # 保存到文件
        torch.save({
            embedding_key: normalized_im_vectors.cpu(), # 保存到 CPU 以减少 GPU 内存占用
            "paths": final_bi_paths
        }, output_filename)

        print(f"Saved {output_filename}")

    print("Embedding generation complete.")