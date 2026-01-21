'''
Evaluation Script for Aircraft (TAC + MGR Only)
=======================================================
Methods Evaluated:
  1. TAC + MGR

Metrics:
  - CLIP Score
  - SigLIP Score
  - DINO v1/v2/v3 (Image Fidelity)
  - KID (Kernel Inception Distance)
  - FID (Frechet Inception Distance)
  - Laplacian Variance (Blurriness)
'''

import os
import sys
import argparse
import cv2

# --------------------------------------------------
# --- 1. Args (Parse BEFORE importing torch) ---
# --------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate Aircraft (TAC + MGR Only)")
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--dinov3_repo_path", type=str, default="/home/tingyu/imageRAG/dinov3")
parser.add_argument("--dinov3_weights_path", type=str, default="/home/tingyu/imageRAG/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
print(f"DEBUG: CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")

import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
import glob

import torch
import torch.nn.functional as F
import torchvision.transforms as T

import open_clip
from open_clip import create_model_from_pretrained, get_tokenizer
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------------------------------
# --- Config ---
# --------------------------------------------------
DATASET_CONFIG = {
    "classes_txt": "datasets/fgvc-aircraft-2013b/data/variants.txt",
    "real_images_list": "datasets/fgvc-aircraft-2013b/data/images_variant_test.txt",
    "real_images_root": "datasets/fgvc-aircraft-2013b/data/images",
}

METHODS = {
    "TAC_MGR": "results/OmniGenV2_TAC_MGR_Aircraft"
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
        # Mock dinov3.data.datasets to avoid ImportError
        import types
        mock_data = types.ModuleType("dinov3.data")
        mock_datasets = types.ModuleType("dinov3.data.datasets")

        # [Fix] Add missing class expected by dinov3
        class DatasetWithEnumeratedTargets: pass
        class SamplerType: pass
        class ImageDataAugmentation: pass
        def make_data_loader(*args, **kwargs): pass

        mock_data.DatasetWithEnumeratedTargets = DatasetWithEnumeratedTargets
        mock_data.SamplerType = SamplerType
        mock_data.ImageDataAugmentation = ImageDataAugmentation
        mock_data.make_data_loader = make_data_loader

        mock_data.datasets = mock_datasets
        sys.modules["dinov3.data"] = mock_data
        sys.modules["dinov3.data.datasets"] = mock_datasets

        # Load model structure from local repo
        models['dino_v3'] = torch.hub.load(args.dinov3_repo_path, 'dinov3_vitb16', source='local', pretrained=False)

        # Load weights from specified path
        print(f"  Loading weights from: {args.dinov3_weights_path}")
        ckpt = torch.load(args.dinov3_weights_path, map_location='cpu')

        # Handle different checkpoint formats (teacher/student/model)
        state_dict = ckpt.get('model', ckpt.get('teacher', ckpt))

        # Clean up state dict keys
        new_state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        models['dino_v3'].load_state_dict(new_state_dict, strict=False)
        models['dino_v3'] = models['dino_v3'].to(device).eval()
    except Exception as e:
        print(f"Error loading DINO v3: {e}")
        # sys.exit(1) # don't exit

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

    # 6. KID & FID (One per method)
    print("Initializing KID & FID...")
    models['kid'] = {}
    models['fid'] = {}
    # Track GroundTruth as well
    ALL_METHODS = list(METHODS.keys()) + ["GroundTruth"] # Modified to match consistent logic
    for m in ALL_METHODS:
        # Reduced subset_size to 5 to allow calculation with fewer samples
        models['kid'][m] = KernelInceptionDistance(subset_size=5, normalize=True).to(device)
        models['fid'][m] = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    models['kid_transform'] = T.Compose([T.Resize(299), T.CenterCrop(299), T.ToTensor()])

    return models

# --------------------------------------------------
# --- 3. Data Loader ---
# --------------------------------------------------
def load_aircraft_tasks(config):
    print("Loading Aircraft Task List...")
    tasks = []
    class_map = {}

    with open(config['classes_txt']) as f:
        for i, l in enumerate(f):
            if l.strip(): class_map[l.strip()] = i

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

# --------------------------------------------------
# --- 4. Evaluation Logic ---
# --------------------------------------------------
def get_best_image_path(method_name, dir_path, safe_filename_prefix):
    # Try to find FINAL, if not, fallback to highest V number
    final_path = os.path.join(dir_path, f"{safe_filename_prefix}_FINAL.png")
    if os.path.exists(final_path):
        return final_path

    # Fallback logic: Find highest V number (up to V10)
    best_v_path = None
    for i in range(10, 1, -1): # Check V10 down to V2
        p = os.path.join(dir_path, f"{safe_filename_prefix}_V1.png")
        if os.path.exists(p):
            best_v_path = p
        if not best_v_path: # Also check V10..V2 if V1 not found? No, loop above does V10..V2.
           pass 
            
    if best_v_path: return best_v_path
    
    v1_p = os.path.join(dir_path, f"{safe_filename_prefix}_V1.png")
    if os.path.exists(v1_p):
        return v1_p

    return None

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

def calculate_laplacian_variance(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None: return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception as e:
        # print(f"Error calculating Laplacian Variance: {e}")
        return 0.0

def evaluate_single(img_path, real_feats_map, txt_feats, models, device, prompt=None):
    scores = {}
    try:
        img = Image.open(img_path).convert("RGB")
        
        # Laplacian Variance
        scores['laplacian_var'] = calculate_laplacian_variance(img_path)

        with torch.no_grad():
            # DINO
            dino_in = models['dino_transform'](img).unsqueeze(0).to(device)
            # v3 might not be loaded if file missing, handle gracefull
            if 'dino_v3' in models:
                out = models['dino_v3'](dino_in)
                out = F.normalize(out, dim=-1)
                scores['dino_v3_base'] = F.cosine_similarity(out, real_feats_map['v3']).mean().item()
            
            # v2
            out = models['dino_v2'](dino_in)
            out = F.normalize(out, dim=-1)
            scores['dino_v2_base'] = F.cosine_similarity(out, real_feats_map['v2']).mean().item()

            # v1
            out = models['dino_v1'](dino_in)
            out = F.normalize(out, dim=-1)
            scores['dino_v1_base'] = F.cosine_similarity(out, real_feats_map['v1']).mean().item()

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

def parse_log_file(file_path):
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        key, val = parts[0].strip(), parts[1].strip()
                        try:
                            val = float(val)
                            if "CLIP" in key: metrics['clip'] = val
                            elif "SigLIP" in key: metrics['siglip'] = val
                            elif "DINO v3" in key: metrics['dino_v3'] = val
                            elif "KID" in key: metrics['kid'] = val
                            elif "FID" in key: metrics['fid'] = val
                            elif "Laplacian" in key: metrics['lap_var'] = val
                        except: pass
        return metrics
    except: return None

# --------------------------------------------------
# --- 5. Main ---
# --------------------------------------------------
def main():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Load Tasks First (Fast)
    tasks = load_aircraft_tasks(DATASET_CONFIG)

    # 2. Check for existing logs
    print(f"\nChecking for existing evaluation logs...")
    cached_results = {}
    methods_to_evaluate = []

    for method_name, dir_path in METHODS.items():
        log_file = os.path.join(dir_path, "logs", "evaluation_metrics.txt")
        if os.path.exists(log_file):
            metrics = parse_log_file(log_file)
            if metrics:
                print(f"  [Cached] Found logs for {method_name}")
                cached_results[method_name] = metrics
                continue
        methods_to_evaluate.append(method_name)

    if not methods_to_evaluate:
        print("\nAll methods have cached results. Skipping evaluation loop.")
    else:
        print(f"\nMethods to evaluate: {methods_to_evaluate}")

    # 3. Evaluation Loop (Only if needed)
    results = {}
    models = {}

    if methods_to_evaluate:
        # Pre-scan only if running eval
        print(f"\nScanning generated images for {len(tasks)} tasks...")
        counts = {m: 0 for m in methods_to_evaluate}
        for task in tasks:
            for method_name in methods_to_evaluate:
                dir_path = METHODS[method_name]
                if get_best_image_path(method_name, dir_path, task['safe_filename_prefix']):
                    counts[method_name] += 1
        
        print(f"  Evaluated Images Count (Total Tasks: {len(tasks)}):")
        for m in methods_to_evaluate:
            print(f"    - {m:<15}: {counts[m]}")
        print("-" * 80)

        models = load_models(device)
        
        # Init results
        results = {m: [] for m in methods_to_evaluate}
        results["GroundTruth"] = []

        print(f"\nStarting evaluation on {len(tasks)} Aircraft classes...")

        for task in tqdm(tasks):
            # [Optimization] Pre-check if any generated image exists for methods to evaluate
            task_img_paths = {}
            for method_name in methods_to_evaluate:
                dir_path = METHODS[method_name]
                p = get_best_image_path(method_name, dir_path, task['safe_filename_prefix'])
                if p:
                    task_img_paths[method_name] = p
            
            # GroundTruth always needs real paths
            real_paths = task["real_image_paths"]
            if not real_paths: continue

            # 1. Get Real Features
            real_feats = {}
            # V3
            if 'dino_v3' in models:
                real_feats['v3'] = get_dino_features_batch(real_paths, models['dino_v3'], models['dino_transform'], device)
            else:
                real_feats['v3'] = None
                
            real_feats['v2'] = get_dino_features_batch(real_paths, models['dino_v2'], models['dino_transform'], device)
            real_feats['v1'] = get_dino_features_batch(real_paths, models['dino_v1'], models['dino_transform'], device)

            if any(v is None for k,v in real_feats.items() if k != 'v3'): 
                continue 

            # 2. Get Text Features
            txt_feats = {}
            with torch.no_grad():
                if models.get('clip_tokenizer'):
                    ctk = models['clip_tokenizer']([task["prompt"]]).to(device)
                    txt_feats['clip'] = models['clip_model'].encode_text(ctk)
                    txt_feats['clip'] /= txt_feats['clip'].norm(dim=-1, keepdim=True)

                # SigLIP
                if models.get('siglip_tokenizer'):
                    try:
                        stk = models['siglip_tokenizer']([task["prompt"]]).to(device)
                    except AttributeError:
                        if hasattr(models['siglip_tokenizer'], 'tokenizer'):
                            internal_tok = models['siglip_tokenizer'].tokenizer
                            if hasattr(internal_tok, '__call__'):
                                res = internal_tok([task["prompt"]], padding='max_length', truncation=True, max_length=64, return_tensors='pt')
                                stk = res['input_ids'].to(device)
                            else: stk = None
                        else: stk = None

                    if stk is not None:
                        txt_feats['siglip'] = models['siglip_model'].encode_text(stk)
                        txt_feats['siglip'] = F.normalize(txt_feats['siglip'], dim=-1)

            # 3. Update KID & FID (Real) with ALL valid real images
            use_paths = real_paths[:50]
            for r_path in use_paths:
                try:
                    kt = models['kid_transform'](Image.open(r_path).convert("RGB")).unsqueeze(0).to(device)
                    # Update only for methods being evaluated + GT
                    for m_key in models['kid'].keys():
                         if m_key in methods_to_evaluate or m_key == "GroundTruth":
                            models['kid'][m_key].update(kt, real=True)
                            models['fid'][m_key].update(kt, real=True)
                except: pass

            # 4. Evaluate Ground Truth
            gt_samples = random.sample(real_paths, min(len(real_paths), 3))
            gt_scores_accum = defaultdict(list)

            for gt_img_path in gt_samples:
                s_res = evaluate_single(gt_img_path, real_feats, txt_feats, models, device, prompt=task["prompt"])
                if s_res:
                    for k, v in s_res.items():
                        gt_scores_accum[k].append(v)
                
                try:
                    img_tensor = models['kid_transform'](Image.open(gt_img_path).convert("RGB")).unsqueeze(0).to(device)
                    models['kid']['GroundTruth'].update(img_tensor, real=False)
                    models['fid']['GroundTruth'].update(img_tensor, real=False)
                except: pass

            if gt_scores_accum:
                gt_final_res = {k: np.mean(v) for k, v in gt_scores_accum.items()}
                results['GroundTruth'].append(gt_final_res)

            # 5. Evaluate Generated Images
            for method_name, img_path in task_img_paths.items():
                s_res = evaluate_single(img_path, real_feats, txt_feats, models, device, prompt=task["prompt"])
                if s_res:
                    results[method_name].append(s_res)
                
                try:
                    img_tensor = models['kid_transform'](Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                    models['kid'][method_name].update(img_tensor, real=False)
                    models['fid'][method_name].update(img_tensor, real=False)
                except: pass

    # Report
    print("\n" + "="*120)
    print(f"  EVAL REPORT: Aircraft (TAC + MGR Only)")
    print("="*120)

    print(f"{'Method':<15} | {'CLIP':<8} | {'SigLIP':<8} | {'DINOv3':<8} | {'KID':<8} | {'FID':<8} | {'LapVar':<8}")
    print("-" * 120)

    # Include GroundTruth in reporting
    all_methods_ordered = ["GroundTruth"] + list(METHODS.keys())

    for method_name in all_methods_ordered:
        # 1. Use Cached if available
        if method_name in cached_results:
            m = cached_results[method_name]
            # Use safe get
            c_s = m.get('clip', 0)
            s_s = m.get('siglip', 0)
            d3 = m.get('dino_v3', 0)
            kv = m.get('kid', 0)
            fv = m.get('fid', 0)
            lv = m.get('lap_var', 0)
            cached_suffix = " (Cached)"
            print(f"{method_name:<15} | {c_s:.4f}   | {s_s:.4f}   | {d3:.4f}   | {kv:.5f} | {fv:.4f} | {lv:.1f}{cached_suffix}")
            continue
        
        # 2. Results from current run
        lst = results.get(method_name, [])
        if not lst:
            if method_name in methods_to_evaluate or method_name == "GroundTruth":
                 print(f"{method_name:<15} | N/A")
            continue

        try:
            kid_score, _ = models['kid'][method_name].compute()
            kid_val = kid_score.item()
        except: kid_val = 0.0

        try:
            fid_val = models['fid'][method_name].compute().item()
        except: fid_val = 0.0

        clip_s = np.mean([x['clip'] for x in lst])
        siglip_s = np.mean([x['siglip'] for x in lst])
        
        if len(lst) > 0:
            if 'dino_v3_base' in lst[0]:
                d3 = np.mean([x['dino_v3_base'] for x in lst])
            else:
                d3 = np.mean([x.get('dinov3_base', 0) for x in lst])
        else: d3 = 0.0

        lap_var = np.mean([x.get('laplacian_var', 0) for x in lst])

        print(f"{method_name:<15} | {clip_s:.4f}   | {siglip_s:.4f}   | {d3:.4f}   | {kid_val:.5f} | {fid_val:.4f} | {lap_var:.1f}")

        # Save to logs
        if method_name in METHODS:
            try:
                dir_path = METHODS[method_name]
                log_dir = os.path.join(dir_path, "logs")
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, "evaluation_metrics.txt")
                
                with open(log_file, "w") as f:
                    f.write(f"Evaluation Metrics for {method_name}\n")
                    f.write("="*50 + "\n")
                    f.write(f"CLIP Score: {clip_s:.4f}\n")
                    f.write(f"SigLIP Score: {siglip_s:.4f}\n")
                    f.write(f"DINO v3 Score: {d3:.4f}\n")
                    f.write(f"KID Score: {kid_val:.5f}\n")
                    f.write(f"FID Score: {fid_val:.4f}\n")
                    f.write(f"Laplacian Variance: {lap_var:.1f}\n")
            except Exception as e:
                print(f"  -> Failed to save logs for {method_name}: {e}")

    print("="*120)

if __name__ == "__main__":
    main()
