'''
Evaluation Script for: OmniGenV2 + BC + MGR (Aircraft)
=======================================================
Metrics:
  - CLIP Score (Text-Image Alignment)
  - SigLIP Score (Better Alignment)
  - DINO v1/v2/v3 (Image Fidelity/Quality) - All ViT-B
  - KID (Kernel Inception Distance) - Distribution Distance
'''

import os
import sys
import argparse
import re
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

# PyTorch & Vision
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# Third-party
import open_clip
from open_clip import create_model_from_pretrained, get_tokenizer
from torchmetrics.image.kid import KernelInceptionDistance

# --------------------------------------------------
# --- 1. Args ---
# --------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate OmniGenV2_BC_MGR_Aircraft")
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=32)

# DINOv3 Paths
parser.add_argument("--dinov3_repo_path", type=str,
                    default="/home/tingyu/imageRAG/dinov3")
parser.add_argument("--dinov3_weights_path", type=str,
                    default="/home/tingyu/imageRAG/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------------------------------
# --- Dataset Config ---
# --------------------------------------------------
DATASET_CONFIG = {
    "classes_txt": "datasets/fgvc-aircraft-2013b/data/variants.txt",
    "real_images_list": "datasets/fgvc-aircraft-2013b/data/images_variant_test.txt",
    "real_images_root": "datasets/fgvc-aircraft-2013b/data/images",
    "results_dir": "results/OmniGenV2_BC_MGR_Aircraft"
}

# --------------------------------------------------
# --- 2. Model Loader ---
# --------------------------------------------------
def load_models(device):
    models = {}
    models['dino_transform'] = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # 1. DINO v3
    print(f"Loading DINO v3...")
    sys.path.append(args.dinov3_repo_path)
    try:
        models['dino_v3'] = torch.hub.load(args.dinov3_repo_path, 'dinov3_vitb16', source='local', pretrained=False)
        ckpt = torch.load(args.dinov3_weights_path, map_location='cpu')
        state_dict = ckpt.get('model', ckpt.get('teacher', ckpt))
        new_state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        models['dino_v3'].load_state_dict(new_state_dict, strict=False)
        models['dino_v3'] = models['dino_v3'].to(device).eval()
    except Exception as e:
        print(f"Error loading DINO v3: {e}")
        sys.exit(1)

    # 2. DINO v2
    print("Loading DINO v2...")
    models['dino_v2'] = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device).eval()

    # 3. DINO v1
    print("Loading DINO v1...")
    models['dino_v1'] = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').to(device).eval()

    # 4. CLIP
    print("Loading CLIP...")
    models['clip_model'], _, models['clip_preprocess'] = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    models['clip_model'] = models['clip_model'].eval().to(device)
    models['clip_tokenizer'] = open_clip.get_tokenizer("ViT-B-32")

    # 5. SigLIP
    print("Loading SigLIP...")
    sig_name = 'hf-hub:timm/ViT-SO400M-14-SigLIP-384'
    models['siglip_model'], models['siglip_preprocess'] = create_model_from_pretrained(sig_name, device=device)
    models['siglip_model'] = models['siglip_model'].eval().to(device)
    models['siglip_tokenizer'] = get_tokenizer(sig_name)

    # 6. KID
    print("Initializing KID...")
    # subset_size=25 allows for 100 samples (4 subsets)
    models['kid_v1'] = KernelInceptionDistance(subset_size=25, normalize=True).to(device)
    models['kid_final'] = KernelInceptionDistance(subset_size=25, normalize=True).to(device)
    models['kid_transform'] = T.Compose([T.Resize(299), T.CenterCrop(299), T.ToTensor()])

    return models

# --------------------------------------------------
# --- 3. Data Loader (Aircraft Specific) ---
# --------------------------------------------------
def load_aircraft_tasks(config):
    print("Loading Aircraft Task List...")
    tasks = []
    class_map = {}

    # Load classes
    with open(config['classes_txt']) as f:
        for i, l in enumerate(f):
            if l.strip(): class_map[l.strip()] = i

    # Load real images (Ground Truth)
    real_map = defaultdict(list)
    with open(config['real_images_list']) as f:
        for l in f:
            p = l.strip().split(' ', 1)
            if len(p) == 2 and p[1] in class_map:
                fp = os.path.join(config['real_images_root'], f"{p[0]}.jpg")
                if os.path.exists(fp):
                    real_map[class_map[p[1]]].append(fp)

    id_to_name = {v: k for k, v in class_map.items()}

    for i in range(len(class_map)):
        name = id_to_name[i]
        safe_name = name.replace(' ', '_').replace('/', '-')
        tasks.append({
            "prompt": f"a photo of a {name}",
            "safe_filename_prefix": f"{safe_name}",
            "real_image_paths": real_map[i]
        })
    return tasks

def find_generated_images(results_dir, prefix):
    v1_path = os.path.join(results_dir, f"{prefix}_V1.png")
    final_path = os.path.join(results_dir, f"{prefix}_FINAL.png")

    if not os.path.exists(v1_path):
        return None, None

    if os.path.exists(final_path):
        return v1_path, final_path

    # Fallback: Find highest V number (up to V10)
    best_path = v1_path
    for i in range(2, 11):
        p = os.path.join(results_dir, f"{prefix}_V{i}.png")
        if os.path.exists(p):
            best_path = p
        else:
            break

    return v1_path, best_path

# --------------------------------------------------
# --- 4. Evaluation Logic ---
# --------------------------------------------------
def get_dino_features_batch(paths, model, transform, device, batch_size=32):
    feats = []
    eval_paths = paths[:50]
    for i in range(0, len(eval_paths), batch_size):
        batch_paths = eval_paths[i:i+batch_size]
        batch_imgs = []
        for p in batch_paths:
            try: batch_imgs.append(transform(Image.open(p).convert("RGB")))
            except: pass
        if not batch_imgs: continue
        batch_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            out = model(batch_tensor)
            feats.append(out)
    if not feats: return None
    return F.normalize(torch.cat(feats, dim=0), dim=-1)

def evaluate_single(img_path, real_feats_map, txt_feats, models, device):
    scores = {}
    try:
        img = Image.open(img_path).convert("RGB")
        with torch.no_grad():
            # DINO
            dino_in = models['dino_transform'](img).unsqueeze(0).to(device)
            for ver in ['v1', 'v2', 'v3']:
                out = models[f'dino_{ver}'](dino_in)
                out = F.normalize(out, dim=-1)
                scores[f'dino{ver}_base'] = F.cosine_similarity(out, real_feats_map[ver]).mean().item()

            # CLIP
            c_in = models['clip_preprocess'](img).unsqueeze(0).to(device)
            c_feat = models['clip_model'].encode_image(c_in)
            c_feat /= c_feat.norm(dim=-1, keepdim=True)
            scores['clip'] = (c_feat @ txt_feats['clip'].T).item()

            # SigLIP
            s_in = models['siglip_preprocess'](img).unsqueeze(0).to(device)
            s_feat = models['siglip_model'].encode_image(s_in)
            s_feat = F.normalize(s_feat, dim=-1)
            scores['siglip'] = (s_feat @ txt_feats['siglip'].T).item()
    except Exception as e:
        print(f"Error: {e}")
        return None
    return scores

# --------------------------------------------------
# --- 5. Main ---
# --------------------------------------------------
def main():
    # Fix Seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    models = load_models(device)
    tasks = load_aircraft_tasks(DATASET_CONFIG)

    scores_base = []
    scores_final = []

    print(f"\nStarting evaluation on {len(tasks)} Aircraft classes...")

    for task in tqdm(tasks):
        v1_path, final_path = find_generated_images(DATASET_CONFIG['results_dir'], task["safe_filename_prefix"])

        if not v1_path: continue

        # 1. Get Real Features
        real_paths = task["real_image_paths"]
        if not real_paths: continue

        real_feats = {}
        for ver in ['v1', 'v2', 'v3']:
            real_feats[ver] = get_dino_features_batch(real_paths, models[f'dino_{ver}'], models['dino_transform'], device)
        if any(v is None for v in real_feats.values()): continue

        # 2. Get Text Features
        txt_feats = {}
        with torch.no_grad():
            ctk = models['clip_tokenizer']([task["prompt"]]).to(device)
            txt_feats['clip'] = models['clip_model'].encode_text(ctk)
            txt_feats['clip'] /= txt_feats['clip'].norm(dim=-1, keepdim=True)

            stk = models['siglip_tokenizer']([task["prompt"]]).to(device)
            txt_feats['siglip'] = models['siglip_model'].encode_text(stk)
            txt_feats['siglip'] = F.normalize(txt_feats['siglip'], dim=-1)

        # 3. Update KID (Real)
        kp = random.sample(real_paths, min(len(real_paths), 20))
        kimgs = []
        for p in kp:
            try: kimgs.append(models['kid_transform'](Image.open(p).convert("RGB")))
            except: pass
        if kimgs:
            kt = torch.stack(kimgs).to(device)
            models['kid_v1'].update(kt, real=True)
            models['kid_final'].update(kt, real=True)

        # 4. Eval V1
        res_v1 = evaluate_single(v1_path, real_feats, txt_feats, models, device)
        if res_v1:
            scores_base.append(res_v1)
            try:
                models['kid_v1'].update(models['kid_transform'](Image.open(v1_path).convert("RGB")).unsqueeze(0).to(device), real=False)
            except Exception as e:
                print(f"Error updating KID V1: {e}")

        # 5. Eval Final
        res_final = evaluate_single(final_path, real_feats, txt_feats, models, device)
        if res_final:
            scores_final.append(res_final)
            try:
                models['kid_final'].update(models['kid_transform'](Image.open(final_path).convert("RGB")).unsqueeze(0).to(device), real=False)
            except Exception as e:
                print(f"Error updating KID Final: {e}")

    # Report
    print("\nComputing KID...")
    try:
        kv1, _ = models['kid_v1'].compute()
    except Exception as e:
        print(f"Warning: KID (V1) computation failed: {e}")
        kv1 = torch.tensor(0.0)

    try:
        kfin, _ = models['kid_final'].compute()
    except Exception as e:
        print(f"Warning: KID (Final) computation failed: {e}")
        kfin = torch.tensor(0.0)

    print("\n" + "="*60)
    print(f"  EVAL REPORT: OmniGenV2 + BC + MGR (Aircraft)")
    print("="*60)

    def show(name, lst, k):
        print(f"\n--- {name} ---")
        if not lst: return
        metrics = ['clip', 'siglip', 'dinov1_base', 'dinov2_base', 'dinov3_base']
        for m in metrics:
            print(f"{m.upper():<12}: {np.mean([x[m] for x in lst]):.4f}")
        print(f"{'KID':<12}: {k.item():.6f}")

    show("Baseline (V1)", scores_base, kv1)
    show("BC+MGR (Final)", scores_final, kfin)
    print("="*60)

if __name__ == "__main__":
    main()
