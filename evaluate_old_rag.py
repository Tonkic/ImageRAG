import os
import sys
import argparse
import glob
import shutil
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import torch
import torch.nn.functional as F
import clip
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
from torchvision import transforms
from torchmetrics.image.kid import KernelInceptionDistance

# ==========================================
# 1. 参数解析 & 显卡设置
# ==========================================
parser = argparse.ArgumentParser(description="Evaluation script for 'old' RAG with DINO v1/v2/v3 (All ViT-B) + KID")
parser.add_argument("--dataset_name", type=str, required=True, choices=['aircraft', 'cub', 'imagenet'])
parser.add_argument("--device_id", type=int, required=True, help="GPU device ID")
parser.add_argument("--ground_truth_root", type=str, default=None, help="Optional: Override config GT root")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction")
parser.add_argument("--clip_model_name", type=str, default="ViT-B/32")
parser.add_argument("--siglip_model_name", type=str, default="ViT-SO400M-14-SigLIP-384")

# --- DINO 配置 (统一 ViT-Base) ---
parser.add_argument("--dino_v1_model_name", type=str, default="dino_vitb16", help="DINO v1 model name (Base)")
parser.add_argument("--dino_v2_model_name", type=str, default="dinov2_vitb14", help="DINO v2 model name (Base)")

# --- DINOv3 本地路径配置 ---
parser.add_argument("--dinov3_repo_path", type=str,
                    default="/home/tingyu/imageRAG/dinov3",
                    help="Path to the cloned dinov3 github repository")
parser.add_argument("--dinov3_weights_path", type=str,
                    default="/home/tingyu/imageRAG/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
                    help="Path to the local .pth weights file for DINOv3 ViT-B/16")

# 解析参数
args = parser.parse_args()

# 设置可见显卡
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# --- 数据库配置中心 ---
# --------------------------------------------------
DATASET_CONFIGS = {
    "aircraft": {
        "classes_txt": "datasets/fgvc-aircraft-2013b/data/variants.txt",
        "output_path": "results/Aircraft_old_RAG",
        "gt_map_files_dir": "datasets/fgvc-aircraft-2013b/data",
        "gt_root": "datasets/fgvc-aircraft-2013b/data/images"
    },
    "cub": {
        "classes_txt": "datasets/CUB_200_2011/classes.txt",
        "output_path": "results/CUB_old_RAG",
        "gt_root": "datasets/CUB_200_2011/images"
    },
    "imagenet": {
        "classes_txt": "datasets/imagenet_classes.txt",
        "train_list": "datasets/imagenet_train_list.txt",
        "output_path": "results/ImageNet_old_RAG",
        "gt_root": "datasets/ILSVRC2012_train"
    }
}

# --------------------------------------------------
# --- 模型加载函数 ---
# --------------------------------------------------
def load_models(device):
    models = {}
    print(f"Loading CLIP model: {args.clip_model_name}")
    models['clip'], models['clip_preprocess'] = clip.load(args.clip_model_name, device=device)

    print(f"Loading SigLIP model: {args.siglip_model_name}")
    models['siglip'], models['siglip_preprocess'] = create_model_from_pretrained(f'hf-hub:timm/{args.siglip_model_name}')
    models['siglip'] = models['siglip'].to(device).eval()
    models['siglip_tokenizer'] = get_tokenizer(f'hf-hub:timm/{args.siglip_model_name}')

    print(f"Loading DINO v1 model: {args.dino_v1_model_name}")
    models['dino_v1'] = torch.hub.load('facebookresearch/dino:main', args.dino_v1_model_name)
    models['dino_v1'] = models['dino_v1'].to(device).eval()

    print(f"Loading DINO v2 model: {args.dino_v2_model_name}")
    models['dino_v2'] = torch.hub.load('facebookresearch/dinov2', args.dino_v2_model_name)
    models['dino_v2'] = models['dino_v2'].to(device).eval()

    print(f"Loading DINO v3 (Local ViT-B/16)...")
    if args.dinov3_repo_path not in sys.path:
        sys.path.append(args.dinov3_repo_path)
    try:
        models['dino_v3'] = torch.hub.load(repo_or_dir=args.dinov3_repo_path, model='dinov3_vitb16', source='local', pretrained=False)
        checkpoint = torch.load(args.dinov3_weights_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint.get('teacher', checkpoint.get('student', checkpoint)))
        new_state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        models['dino_v3'].load_state_dict(new_state_dict, strict=False)
        models['dino_v3'] = models['dino_v3'].to(device).eval()
    except Exception as e:
        print(f"Error loading DINOv3: {e}"); sys.exit(1)

    models['dino_preprocess'] = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    print("Initializing KID Metrics...")
    models['kid_transform'] = transforms.Compose([transforms.Resize(299), transforms.CenterCrop(299), transforms.ToTensor()])
    models['kid_baseline'] = KernelInceptionDistance(subset_size=50, normalize=True).to(device)
    models['kid_rag'] = KernelInceptionDistance(subset_size=50, normalize=True).to(device)
    return models

# --------------------------------------------------
# --- 特征提取辅助函数 ---
# --------------------------------------------------
def get_text_features(model, text, device, tokenizer=None):
    with torch.no_grad():
        if tokenizer:
            tokens = tokenizer(text).to(device)
            features = model.encode_text(tokens)
        else:
            tokens = clip.tokenize(text).to(device)
            features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
    return features

def get_image_features_batch(image_paths, model, preprocess, device, batch_size=32):
    all_features = []
    eval_paths = image_paths[:100]
    for i in range(0, len(eval_paths), batch_size):
        batch_paths = eval_paths[i:i+batch_size]
        batch_imgs = []
        for p in batch_paths:
            try: img = Image.open(p).convert("RGB"); batch_imgs.append(preprocess(img))
            except: pass
        if not batch_imgs: continue
        batch_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            model_name = model.__class__.__name__
            if 'CLIP' == model_name: feats = model.encode_image(batch_tensor)
            elif 'Siglip' in model_name: feats = model.encode_image(batch_tensor)
            else: feats = model(batch_tensor)
            all_features.append(feats)
    if not all_features: return None
    all_features = torch.cat(all_features, dim=0)
    return F.normalize(all_features, dim=-1)

# --------------------------------------------------
# --- GT Map 构建 ---
# --------------------------------------------------
def build_gt_map(dataset_name, config):
    mapping = defaultdict(list)
    if dataset_name == 'aircraft':
        gt_dir = config["gt_map_files_dir"]
        for fname in ["images_variant_train.txt", "images_variant_val.txt", "images_variant_test.txt"]:
            fpath = os.path.join(gt_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, 'r') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2: mapping[parts[1]].append(parts[0])
    elif dataset_name == 'imagenet':
        with open(config["train_list"], 'r') as f:
            for line in f:
                path = line.strip()
                if path: mapping[path.split('/')[0]].append(path)
    return mapping

# --------------------------------------------------
# --- 主逻辑 ---
# --------------------------------------------------
if __name__ == "__main__":
    config = DATASET_CONFIGS[args.dataset_name]
    results_dir = config["output_path"]
    gt_root = args.ground_truth_root if args.ground_truth_root else config.get("gt_root")

    if not gt_root or not os.path.exists(gt_root):
        print(f"Error: GT root {gt_root} not found.")
        sys.exit(1)

    models = load_models(device)

    all_items_to_generate = []
    with open(config["classes_txt"], 'r') as f:
        for i, line in enumerate(f.readlines(), 0):
            line = line.strip()
            if not line: continue
            if args.dataset_name == 'imagenet':
                parts = line.split(':', 1)
                all_items_to_generate.append((i, parts[1].split(',')[0].strip(), parts[0].strip()))
            elif args.dataset_name == 'cub':
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    folder_name = parts[1]
                    simple_name = folder_name.split('.', 1)[-1].replace('_', ' ')
                    all_items_to_generate.append((i, simple_name, folder_name))
            elif args.dataset_name == 'aircraft':
                all_items_to_generate.append((i, line, line))

    gt_map = build_gt_map(args.dataset_name, config)

    metrics_keys = ['clip', 'siglip', 'dino_v1_base', 'dino_v2_base', 'dino_v3_base']
    baseline_scores = {k: [] for k in metrics_keys}
    rag_scores = {k: [] for k in metrics_keys}

    valid_class_count = 0

    # === 诊断记录列表 ===
    missing_baselines_list = []
    missing_gt_list = []

    print(f"\n--- Starting Evaluation for {args.dataset_name} ---")
    print(f"Target Total Classes: {len(all_items_to_generate)}")

    for label_id, simple_name, lookup_key in tqdm(all_items_to_generate):
        prompt = f"a photo of a {simple_name}"

        # --- Step 1: 寻找 Baseline 图片 (No RAG) ---
        search_pattern = os.path.join(results_dir, f"{label_id:03d}_*_no_imageRAG.png")
        found_baselines = glob.glob(search_pattern)

        if not found_baselines:
            # [DIAGNOSTIC] 记录缺失的 Baseline
            missing_baselines_list.append(f"ID {label_id:03d}: {simple_name}")
            # print(f"[DEBUG MISSING BASELINE] ID {label_id:03d}") # 保持进度条整洁，最后统一打印
            continue

        baseline_path = found_baselines[0]

        # --- Step 2: 寻找 RAG 图片 (Fallback Logic) ---
        base_filename = os.path.basename(baseline_path)
        file_prefix = base_filename.replace("_no_imageRAG.png", "")
        rag_pattern = os.path.join(results_dir, f"{file_prefix}_gs_*.png")
        rag_candidates = glob.glob(rag_pattern)

        final_rag_path = rag_candidates[0] if rag_candidates else baseline_path

        # --- Step 3: 获取 GT 图片列表 ---
        gt_paths = []
        if args.dataset_name == 'imagenet':
            rel_paths = gt_map.get(lookup_key, [])
            gt_paths = [os.path.join(gt_root, p) for p in rel_paths]
        elif args.dataset_name == 'aircraft':
            ids = gt_map.get(lookup_key, [])
            gt_paths = [os.path.join(gt_root, f"{id}.jpg") for id in ids]
        elif args.dataset_name == 'cub':
            gt_dir = os.path.join(gt_root, lookup_key)
            if os.path.exists(gt_dir):
                gt_paths = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith(('.jpg', '.png'))]

        gt_paths = [p for p in gt_paths if os.path.exists(p)]

        if not gt_paths:
            # [DIAGNOSTIC] 记录缺失的 GT
            missing_gt_list.append(f"ID {label_id:03d}: {lookup_key}")
            continue

        valid_class_count += 1

        # --- Step 4: 提取特征和计算分数 ---
        try:
            clip_t = get_text_features(models['clip'], [prompt], device)
            siglip_t = get_text_features(models['siglip'], [prompt], device, models['siglip_tokenizer'])

            gt_v1 = get_image_features_batch(gt_paths, models['dino_v1'], models['dino_preprocess'], device)
            gt_v2 = get_image_features_batch(gt_paths, models['dino_v2'], models['dino_preprocess'], device)
            gt_v3 = get_image_features_batch(gt_paths, models['dino_v3'], models['dino_preprocess'], device)

            if any(x is None for x in [gt_v1, gt_v2, gt_v3]):
                print(f"[Feature Error] ID {label_id:03d} feature extraction failed.")
                continue

            base_pil = Image.open(baseline_path).convert("RGB")
            rag_pil = Image.open(final_rag_path).convert("RGB")

            # KID
            import random
            kid_real_samples = random.sample(gt_paths, min(len(gt_paths), 50))
            kid_real_imgs = [models['kid_transform'](Image.open(p).convert("RGB")) for p in kid_real_samples]
            if kid_real_imgs:
                kid_real_tensor = torch.stack(kid_real_imgs).to(device)
                models['kid_baseline'].update(kid_real_tensor, real=True)
                models['kid_rag'].update(kid_real_tensor, real=True)

            base_tensor_kid = models['kid_transform'](base_pil).unsqueeze(0).to(device)
            rag_tensor_kid = models['kid_transform'](rag_pil).unsqueeze(0).to(device)
            models['kid_baseline'].update(base_tensor_kid, real=False)
            models['kid_rag'].update(rag_tensor_kid, real=False)

            # Features
            dino_in_base = models['dino_preprocess'](base_pil).unsqueeze(0).to(device)
            dino_in_rag = models['dino_preprocess'](rag_pil).unsqueeze(0).to(device)
            clip_in_base = models['clip_preprocess'](base_pil).unsqueeze(0).to(device)
            clip_in_rag = models['clip_preprocess'](rag_pil).unsqueeze(0).to(device)
            sig_in_base = models['siglip_preprocess'](base_pil).unsqueeze(0).to(device)
            sig_in_rag = models['siglip_preprocess'](rag_pil).unsqueeze(0).to(device)

            # Scoring
            base_clip = models['clip'].encode_image(clip_in_base)
            base_clip /= base_clip.norm(dim=-1, keepdim=True)
            rag_clip = models['clip'].encode_image(clip_in_rag)
            rag_clip /= rag_clip.norm(dim=-1, keepdim=True)
            baseline_scores['clip'].append((base_clip @ clip_t.T).item())
            rag_scores['clip'].append((rag_clip @ clip_t.T).item())

            base_sig = F.normalize(models['siglip'].encode_image(sig_in_base), dim=-1)
            rag_sig = F.normalize(models['siglip'].encode_image(sig_in_rag), dim=-1)
            baseline_scores['siglip'].append((base_sig @ siglip_t.T).item())
            rag_scores['siglip'].append((rag_sig @ siglip_t.T).item())

            base_v1 = F.normalize(models['dino_v1'](dino_in_base), dim=-1)
            rag_v1 = F.normalize(models['dino_v1'](dino_in_rag), dim=-1)
            baseline_scores['dino_v1_base'].append((base_v1 @ gt_v1.T).mean().item())
            rag_scores['dino_v1_base'].append((rag_v1 @ gt_v1.T).mean().item())

            base_v2 = F.normalize(models['dino_v2'](dino_in_base), dim=-1)
            rag_v2 = F.normalize(models['dino_v2'](dino_in_rag), dim=-1)
            baseline_scores['dino_v2_base'].append((base_v2 @ gt_v2.T).mean().item())
            rag_scores['dino_v2_base'].append((rag_v2 @ gt_v2.T).mean().item())

            base_v3 = F.normalize(models['dino_v3'](dino_in_base), dim=-1)
            rag_v3 = F.normalize(models['dino_v3'](dino_in_rag), dim=-1)
            baseline_scores['dino_v3_base'].append((base_v3 @ gt_v3.T).mean().item())
            rag_scores['dino_v3_base'].append((rag_v3 @ gt_v3.T).mean().item())

        except Exception as e:
            print(f"Error processing {simple_name}: {e}")

    print("Calculating Final KID Scores...")
    try:
        kid_base_mean, _ = models['kid_baseline'].compute()
        kid_rag_mean, _ = models['kid_rag'].compute()
    except:
        kid_base_mean = torch.tensor(0.0); kid_rag_mean = torch.tensor(0.0)

    print("\n\n" + "="*60)
    print(f"  EVALUATION RESULTS: {args.dataset_name} (All ViT-Base)")
    print(f"  Classes Evaluated: {len(baseline_scores['clip'])} / {len(all_items_to_generate)}")
    print("="*60)

    # --- 打印缺失列表 ---
    if missing_baselines_list:
        print("\n[WARNING] The following classes are MISSING generated images (no_imageRAG):")
        for item in missing_baselines_list:
            print(f"  - {item}")
        print("  -> Solution: Please run the generation script for these specific IDs.")

    if missing_gt_list:
        print("\n[WARNING] The following classes are MISSING Ground Truth images:")
        for item in missing_gt_list:
            print(f"  - {item}")
        print("  -> Solution: Check your CUB dataset path or classes.txt mapping.")

    def print_metrics(name, scores, kid_val):
        print(f"\n--- {name} ---")
        if scores['clip']:
            print(f"{'CLIP':<12}: {np.mean(scores['clip']):.4f}")
            print(f"{'SigLIP':<12}: {np.mean(scores['siglip']):.4f}")
            print(f"{'DINOv1_B':<12}: {np.mean(scores['dino_v1_base']):.4f}")
            print(f"{'DINOv2_B':<12}: {np.mean(scores['dino_v2_base']):.4f}")
            print(f"{'DINOv3_B':<12}: {np.mean(scores['dino_v3_base']):.4f}")
            print(f"{'KID':<12}: {kid_val.item():.6f} (Lower is better)")
        else:
            print("No data.")

    print_metrics("Baseline (No RAG)", baseline_scores, kid_base_mean)
    print_metrics("RAG Assisted (Old - Final)", rag_scores, kid_rag_mean)
    print("="*60)