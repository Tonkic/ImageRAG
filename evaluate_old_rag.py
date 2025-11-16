'''
python evaluate_old_rag.py \
    --dataset_name aircraft \
    --device_id 0 \
    --ground_truth_root datasets/fgvc-aircraft-2013b/data/images
'''


import argparse
import sys
import os
import glob
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# --- 1. ！！！关键：立即解析参数！！！ ---
parser = argparse.ArgumentParser(description="Evaluation script for 'old' RAG (Baseline vs RAG)")
parser.add_argument("--dataset_name", type=str, required=True, choices=['aircraft', 'cub'])
parser.add_argument("--device_id", type=int, required=True, help="GPU device ID")
parser.add_argument("--ground_truth_root", type=str, required=True, help="Path to the root of ground truth images (e.g., .../CUB_200_2011/images)")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction")
parser.add_argument("--clip_model_name", type=str, default="ViT-B/32")
parser.add_argument("--siglip_model_name", type=str, default="ViT-SO400M-14-SigLIP-384")
parser.add_argument("--dino_model_name", type=str, default="dinov2_vitb14")

args = parser.parse_args()

# --- 2. ！！！关键：立即设置环境！！！ ---
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

# --- 3. ！！！现在才导入 torch 和其他库！！！ ---
import torch
import clip
from open_clip import create_model_from_pretrained, get_tokenizer
# import timm # <-- 不再需要
from PIL import Image
from torchvision import transforms # <-- 关键修复：导入 transforms

# --------------------------------------------------
# --- 数据库配置中心 (与 imageRAG_OmniGen2.py 匹配) ---
# --------------------------------------------------
DATASET_CONFIGS = {
    "aircraft": {
        "classes_txt": "datasets/fgvc-aircraft-2013b/data/variants.txt",
        "output_path": "results/Aircraft_SmartRAG_old_retrieval",
        # ！！！关键修复：添加 aircraft GT 映射文件路径！！！
        "gt_map_files_dir": "datasets/fgvc-aircraft-2013b/data"
    },
    "cub": {
        "classes_txt": "datasets/CUB_200_2011/classes.txt",
        "output_path": "results/CUB_SmartRAG_old_retrieval"
    }
}

# --------------------------------------------------
# --- 辅助函数：模型加载 ---
# --------------------------------------------------
def load_clip_model(device):
    print(f"Loading CLIP model: {args.clip_model_name}")
    model, preprocess = clip.load(args.clip_model_name, device=device)
    return model, preprocess

def load_siglip_model(device):
    print(f"Loading SigLIP model: {args.siglip_model_name}")
    model, preprocess = create_model_from_pretrained(f'hf-hub:timm/{args.siglip_model_name}')
    model = model.to(device)
    tokenizer = get_tokenizer(f'hf-hub:timm/{args.siglip_model_name}')
    return model, preprocess, tokenizer

def load_dino_model(device):
    print(f"Loading DINO V2 model: {args.dino_model_name} (from torch.hub)")
    model = torch.hub.load('facebookresearch/dinov2', args.dino_model_name)
    model = model.to(device)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return model, preprocess

# --------------------------------------------------
# --- 辅助函数：特征提取 ---
# --------------------------------------------------

def batch_images(images, preprocess, batch_size):
    """ 将图像列表分批处理 """
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        processed_batch = torch.stack([preprocess(img) for img in batch])
        yield processed_batch

def get_image_features(model, images, preprocess, device, batch_size=32):
    all_features = []
    with torch.no_grad():
        for image_batch in batch_images(images, preprocess, batch_size):
            image_batch = image_batch.to(device)

            model_class_name = model.__class__.__name__

            if model_class_name == 'CLIP':
                features = model.encode_image(image_batch)
            elif 'Siglip' in model_class_name: # e.g., SiglipVisionTransformer
                features = model.encode_image(image_batch)
            elif 'Dino' in model_class_name: # e.g., DinoVisionTransformer
                features = model(image_batch) # DINO V2 (torch.hub) 返回 CLS token
            else:
                # 备用逻辑
                try:
                    features = model.encode_image(image_batch)
                except AttributeError:
                    features = model(image_batch)

            all_features.append(features.cpu())

    all_features = torch.cat(all_features, dim=0)
    all_features /= all_features.norm(dim=-1, keepdim=True)
    return all_features

def get_text_features(model, text, device, tokenizer=None):
    with torch.no_grad():
        if tokenizer: # SigLIP
            tokens = tokenizer(text).to(device)
            features = model.encode_text(tokens)
        else: # CLIP
            tokens = clip.tokenize(text).to(device)
            features = model.encode_text(tokens)

        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu()

def calculate_similarity(feat1, feat2):
    """ 计算两个归一化特征张量之间的余弦相似度 """
    return torch.matmul(feat1, feat2.T)

# --------------------------------------------------
# --- ！！！关键修复：Aircraft GT 映射函数！！！ ---
# --------------------------------------------------
def build_aircraft_gt_map(gt_map_dir):
    """
    读取 Aircarft 的 'images_variant_*.txt' 文件，
    返回一个字典 {"class_name": ["image_id_1", "image_id_2", ...]}
    """
    print(f"Building Aircraft GT map from: {gt_map_dir}")
    mapping = defaultdict(list)
    files_to_read = [
        "images_variant_train.txt",
        "images_variant_val.txt",
        "images_variant_test.txt"
    ]

    for file_name in files_to_read:
        file_path = os.path.join(gt_map_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: Cannot find map file {file_path}. DINO scores may be inaccurate.")
            continue

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    image_id, class_name = parts
                    mapping[class_name].append(image_id)

    print(f"Aircraft GT map built. Found {len(mapping)} classes.")
    return mapping
# --------------------------------------------------
# (修复结束)
# --------------------------------------------------


# --------------------------------------------------
# --- 主评估逻辑 ---
# --------------------------------------------------

if __name__ == "__main__":
    device = "cuda"
    config = DATASET_CONFIGS[args.dataset_name]

    results_dir = config["output_path"]
    classes_txt = config["classes_txt"]
    gt_root = args.ground_truth_root

    if not os.path.exists(gt_root):
        print(f"Error: Ground truth root directory not found: {gt_root}")
        sys.exit(1)

    # 1. 加载所有模型
    clip_model, clip_preprocess = load_clip_model(device)
    siglip_model, siglip_preprocess, siglip_tokenizer = load_siglip_model(device)
    dino_model, dino_preprocess = load_dino_model(device)

    # 2. 加载类别列表 (与 imageRAG_OmniGen2.py 完全相同)
    print(f"Loading class list from {classes_txt}...")
    all_items_to_generate = []
    try:
        with open(classes_txt) as f:
            for i, line in enumerate(f.readlines()):
                full_class_name = line.strip() # CUB: "001.Black_footed_Albatross", Aircraft: "707-320"
                if not full_class_name: continue

                simple_name = full_class_name # 默认为 aircraft
                if args.dataset_name == 'cub':
                    simple_name = full_class_name.split('.', 1)[-1].replace('_', ' ')

                # (label_id, prompt_name, ground_truth_folder_name)
                all_items_to_generate.append((i, simple_name, full_class_name))
    except FileNotFoundError:
        print(f"Error: Could not find class file {classes_txt}.")
        sys.exit(1)

    print(f"Found {len(all_items_to_generate)} classes to evaluate.")

    # --------------------------------------------------
    # --- ！！！关键修复：预加载 Aircraft GT 映射！！！ ---
    # --------------------------------------------------
    aircraft_gt_map = None
    if args.dataset_name == 'aircraft':
        aircraft_gt_map = build_aircraft_gt_map(config["gt_map_files_dir"])
    # --------------------------------------------------

    # 3. 初始化分数聚合器
    baseline_scores = {'clip': [], 'siglip': [], 'dino': []}
    rag_scores = {'clip': [], 'siglip': [], 'dino': []}

    # 4. 遍历所有类别
    for label_id, simple_name, full_class_name in tqdm(all_items_to_generate, desc="Evaluating classes"):

        current_prompt = f"a photo of a {simple_name}"
        safe_class_name = simple_name.replace(' ', '_').replace('/', '_')
        current_out_name = f"{label_id:03d}_{safe_class_name}"

        # --- A. 查找生成的图像 ---
        baseline_img_path = os.path.join(results_dir, f"{current_out_name}_no_imageRAG.png")
        rag_img_glob = os.path.join(results_dir, f"{current_out_name}_gs_*.png")
        rag_files_found = glob.glob(rag_img_glob)

        if not os.path.exists(baseline_img_path):
            # (打印一次警告，然后继续)
            if label_id < 5: # 仅打印前几个错误
                 print(f"\n[Skip] Missing baseline image: {baseline_img_path}")
            continue
        if not rag_files_found:
            if label_id < 5:
                 print(f"\n[Skip] Missing RAG image (glob failed): {rag_img_glob}")
            continue

        rag_img_path = rag_files_found[0]

        # --------------------------------------------------
        # --- ！！！关键修复：动态 GT 路径加载！！！ ---
        # --------------------------------------------------
        gt_image_paths = []
        if args.dataset_name == 'cub':
            gt_class_dir = os.path.join(gt_root, full_class_name)
            if not os.path.exists(gt_class_dir):
                if label_id < 5:
                    print(f"\n[Skip] Missing CUB Ground Truth directory: {gt_class_dir}")
                continue
            gt_image_paths = [os.path.join(gt_class_dir, f) for f in os.listdir(gt_class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        elif args.dataset_name == 'aircraft':
            if aircraft_gt_map and full_class_name in aircraft_gt_map:
                image_ids = aircraft_gt_map[full_class_name]
                # gt_root 此时是 '.../data/images'
                gt_image_paths = [os.path.join(gt_root, f"{img_id}.jpg") for img_id in image_ids]
            else:
                if label_id < 5:
                    print(f"\n[Skip] Class '{full_class_name}' not found in Aircraft GT map.")
                continue
        # --------------------------------------------------

        try:
            if not gt_image_paths:
                if label_id < 5:
                    print(f"\n[Skip] No GT images found for class: {full_class_name}")
                continue

            baseline_pil = Image.open(baseline_img_path).convert("RGB")
            rag_pil = Image.open(rag_img_path).convert("RGB")
            gt_pils = [Image.open(p).convert("RGB") for p in gt_image_paths]

        except Exception as e:
            print(f"\n[Error] Failed to load images for {current_out_name}: {e}")
            continue

        # --- C. 计算特征 ---

        # 文本特征 (T-Feats)
        clip_t_feat = get_text_features(clip_model, [current_prompt], device)
        siglip_t_feat = get_text_features(siglip_model, [current_prompt], device, siglip_tokenizer)

        # Ground Truth 图像特征 (GT-I-Feats)
        gt_clip_i_feats = get_image_features(clip_model, gt_pils, clip_preprocess, device, args.batch_size)
        gt_siglip_i_feats = get_image_features(siglip_model, gt_pils, siglip_preprocess, device, args.batch_size)
        gt_dino_i_feats = get_image_features(dino_model, gt_pils, dino_preprocess, device, args.batch_size)

        # 平均 GT 特征
        gt_clip_i_feat_avg = torch.mean(gt_clip_i_feats, dim=0, keepdim=True)
        gt_siglip_i_feat_avg = torch.mean(gt_siglip_i_feats, dim=0, keepdim=True)
        gt_dino_i_feat_avg = torch.mean(gt_dino_i_feats, dim=0, keepdim=True)

        # Baseline 图像特征 (Gen-I-Feats)
        baseline_clip_i_feat = get_image_features(clip_model, [baseline_pil], clip_preprocess, device)
        baseline_siglip_i_feat = get_image_features(siglip_model, [baseline_pil], siglip_preprocess, device)
        baseline_dino_i_feat = get_image_features(dino_model, [baseline_pil], dino_preprocess, device)

        # RAG 图像特征
        rag_clip_i_feat = get_image_features(clip_model, [rag_pil], clip_preprocess, device)
        rag_siglip_i_feat = get_image_features(siglip_model, [rag_pil], siglip_preprocess, device)
        rag_dino_i_feat = get_image_features(dino_model, [rag_pil], dino_preprocess, device)

        # --- D. 计算分数并聚合 ---

        # Baseline 分数
        baseline_scores['clip'].append(calculate_similarity(clip_t_feat, baseline_clip_i_feat).item())
        baseline_scores['siglip'].append(calculate_similarity(siglip_t_feat, baseline_siglip_i_feat).item())
        baseline_scores['dino'].append(calculate_similarity(gt_dino_i_feat_avg, baseline_dino_i_feat).item())

        # RAG 分数
        rag_scores['clip'].append(calculate_similarity(clip_t_feat, rag_clip_i_feat).item())
        rag_scores['siglip'].append(calculate_similarity(siglip_t_feat, rag_siglip_i_feat).item())
        rag_scores['dino'].append(calculate_similarity(gt_dino_i_feat_avg, rag_dino_i_feat).item())

    # --- 5. 打印最终结果 ---

    print("\n\n" + "="*30)
    print(f"  EVALUATION RESULTS: {args.dataset_name}")
    print(f"  (Evaluated {len(baseline_scores['clip'])} / {len(all_items_to_generate)} total classes)")
    print("="*30)

    print(f"\n--- Baseline (No RAG) ---")
    for metric, values in baseline_scores.items():
        if values:
            print(f"{metric.upper()} Score : \t {np.mean(values):.4f}")
        else:
            print(f"{metric.upper()} Score : \t N/A")

    print(f"\n--- RAG Assisted (Old) ---")
    for metric, values in rag_scores.items():
        if values:
            print(f"{metric.upper()} Score : \t {np.mean(values):.4f}")
        else:
            print(f"{metric.upper()} Score : \t N/A")

    print("="*30)