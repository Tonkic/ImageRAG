'''
Evaluation Script for ImageNet (All Methods)
=======================================================
Methods Evaluated:
  1. Baseline (V1) - Shared
  2. BC + SR
  3. BC + MGR
  4. TAC + SR
  5. TAC + MGR

Metrics:
  - CLIP Score
  - SigLIP Score
  - DINO v1/v2/v3 (Image Fidelity)
  - KID (Kernel Inception Distance)
'''

import os
import sys
import argparse
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision.transforms as T

import open_clip
from open_clip import create_model_from_pretrained, get_tokenizer
from torchmetrics.image.kid import KernelInceptionDistance

# --------------------------------------------------
# --- 1. Args ---
# --------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate ImageNet (All Methods)")
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--dinov3_repo_path", type=str, default="/home/tingyu/imageRAG/dinov3")
parser.add_argument("--dinov3_weights_path", type=str, default="/home/tingyu/imageRAG/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------------------------------
# --- Config ---
# --------------------------------------------------
DATASET_CONFIG = {
    "classes_txt": "datasets/imagenet_classes.txt",
    "real_images_root": "datasets/ILSVRC2012_train",
}

METHODS = {
    "Baseline": "results/OmniGenV2_Baseline_ImageNet",
    "BC_SR": "results/OmniGenV2_BC_SR_ImageNet",
    "BC_MGR": "results/OmniGenV2_BC_MGR_ImageNet",
    "TAC_SR": "results/OmniGenV2_TAC_SR_ImageNet",
    "TAC_MGR": "results/OmniGenV2_TAC_MGR_ImageNet"
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

    # 6. KID (One per method)
    print("Initializing KID...")
    models['kid'] = {}
    for m in METHODS.keys():
        models['kid'][m] = KernelInceptionDistance(subset_size=25, normalize=True).to(device)

    models['kid_transform'] = T.Compose([T.Resize(299), T.CenterCrop(299), T.ToTensor()])

    return models

# --------------------------------------------------
# --- 3. Data Loader ---
# --------------------------------------------------
def load_imagenet_tasks(config):
    print("Loading ImageNet Task List...")
    tasks = []

    class_info = {}
    with open(config['classes_txt']) as f:
        for line in f:
            parts = line.strip().split(':', 1)
            if len(parts) < 2: continue
            nid = parts[0].strip()
            names = parts[1].strip()
            simple_name = names.split(',')[0].strip()
            class_info[nid] = simple_name

    for nid, simple_name in class_info.items():
        safe_name = f"{nid}_{simple_name.replace(' ', '_').replace('/', '-')}"

        class_dir = os.path.join(config['real_images_root'], nid)
        real_paths = []
        if os.path.isdir(class_dir):
             for img in os.listdir(class_dir):
                 if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                     real_paths.append(os.path.join(class_dir, img))

        tasks.append({
            "prompt": f"a photo of a {simple_name}",
            "safe_filename_prefix": safe_name,
            "real_image_paths": real_paths
        })

    return tasks

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
        # print(f"Error evaluating {img_path}: {e}")
        return None
    return scores

# --------------------------------------------------
# --- 5. Main ---
# --------------------------------------------------
def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    models = load_models(device)
    tasks = load_imagenet_tasks(DATASET_CONFIG)

    # Storage for results
    results = {m: [] for m in METHODS.keys()}

    print(f"\nStarting evaluation on {len(tasks)} ImageNet classes...")

    for task in tqdm(tasks):
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

        # 3. Update KID (Real) - Update ALL KID models with real data
        kp = random.sample(real_paths, min(len(real_paths), 20))
        kimgs = []
        for p in kp:
            try: kimgs.append(models['kid_transform'](Image.open(p).convert("RGB")))
            except: pass

        if kimgs:
            kt = torch.stack(kimgs).to(device)
            for m in METHODS.keys():
                models['kid'][m].update(kt, real=True)

        # 4. Evaluate Each Method
        for method_name, dir_path in METHODS.items():
            # Determine file path
            if method_name == "Baseline":
                img_path = os.path.join(dir_path, f"{task['safe_filename_prefix']}_V1.png")
            else:
                img_path = os.path.join(dir_path, f"{task['safe_filename_prefix']}_FINAL.png")

            if not os.path.exists(img_path):
                continue

            # Eval
            res = evaluate_single(img_path, real_feats, txt_feats, models, device)
            if res:
                results[method_name].append(res)
                try:
                    models['kid'][method_name].update(
                        models['kid_transform'](Image.open(img_path).convert("RGB")).unsqueeze(0).to(device),
                        real=False
                    )
                except: pass

    # Report
    print("\n" + "="*80)
    print(f"  EVAL REPORT: ImageNet (All Methods)")
    print("="*80)
    print(f"{'Method':<15} | {'CLIP':<8} | {'SigLIP':<8} | {'DINOv1':<8} | {'DINOv2':<8} | {'DINOv3':<8} | {'KID':<8}")
    print("-" * 80)

    for method_name in METHODS.keys():
        lst = results[method_name]
        if not lst:
            print(f"{method_name:<15} | N/A")
            continue

        # Compute KID
        try:
            kid_score, _ = models['kid'][method_name].compute()
            kid_val = kid_score.item()
        except:
            kid_val = 0.0

        clip_s = np.mean([x['clip'] for x in lst])
        siglip_s = np.mean([x['siglip'] for x in lst])
        d1 = np.mean([x['dinov1_base'] for x in lst])
        d2 = np.mean([x['dinov2_base'] for x in lst])
        d3 = np.mean([x['dinov3_base'] for x in lst])

        print(f"{method_name:<15} | {clip_s:.4f}   | {siglip_s:.4f}   | {d1:.4f}   | {d2:.4f}   | {d3:.4f}   | {kid_val:.5f}")

    print("="*80)

if __name__ == "__main__":
    main()
