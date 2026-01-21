'''
Recursive Evaluation Script for All Results
=======================================================
Recursively scans /home/tingyu/imageRAG/results for any experiment folder.
Experiments are identified by checking if they contain generated images
(like *_FINAL.png or *_V*.png).

Behavior:
1. Scan recursively for experiment folders.
2. Check for "logs/evaluation_metrics.txt".
   - If found: Load cached metrics.
   - If missing: Add to evaluation queue.
3. Run evaluation on missing experiments.
4. Save logs to each experiment folder.
5. Print a consolidated BIG TABLE of all results.

Metrics Evaluated:
  - CLIP Score
  - SigLIP Score
  - DINO v3 (Image Fidelity)
  - KID
  - FID
  - Laplacian Variance
'''

import os
import sys
import argparse
import cv2
import glob
from collections import defaultdict
import random
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import open_clip
from open_clip import create_model_from_pretrained, get_tokenizer
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# --------------------------------------------------
# --- 1. Args ---
# --------------------------------------------------
parser = argparse.ArgumentParser(description="Recursive Evaluate All Results")
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/home/tingyu/imageRAG/results")
# Use Aircraft dataset config by default as requested context seems to imply
parser.add_argument("--classes_txt", type=str, default="datasets/fgvc-aircraft-2013b/data/variants.txt")
parser.add_argument("--real_images_list", type=str, default="datasets/fgvc-aircraft-2013b/data/images_variant_test.txt")
parser.add_argument("--real_images_root", type=str, default="datasets/fgvc-aircraft-2013b/data/images")
parser.add_argument("--dinov3_repo_path", type=str, default="/home/tingyu/imageRAG/dinov3")
parser.add_argument("--dinov3_weights_path", type=str, default="/home/tingyu/imageRAG/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------------------------------
# --- 2. Utils & Discovery ---
# --------------------------------------------------

def is_experiment_folder(path):
    # Heuristic: Contains generated images?
    # Look for *_FINAL.png or *_V1.png
    if not os.path.isdir(path): return False

    has_final = len(glob.glob(os.path.join(path, "*_FINAL.png"))) > 0
    has_ver = len(glob.glob(os.path.join(path, "*_V1.png"))) > 0

    return has_final or has_ver

def discover_experiments(root):
    experiments = []
    for root_path, dirs, files in os.walk(root):
        if is_experiment_folder(root_path):
            experiments.append(root_path)
    return experiments

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
                            elif "IS" in key: metrics['inception_score'] = val
                            elif "Laplacian" in key: metrics['lap_var'] = val
                        except: pass
        return metrics
    except: return None

# --------------------------------------------------
# --- 3. Dataset & Task Logic ---
# --------------------------------------------------
def load_aircraft_tasks():
    print("Loading Aircraft Task List...")
    tasks = []
    class_map = {}

    if not os.path.exists(args.classes_txt):
        print(f"Dataset config not found: {args.classes_txt}")
        sys.exit(1)

    with open(args.classes_txt) as f:
        for i, l in enumerate(f):
            if l.strip(): class_map[l.strip()] = i

    real_map = defaultdict(list)
    with open(args.real_images_list) as f:
        for l in f:
            p = l.strip().split(' ', 1)
            if len(p) == 2 and p[1] in class_map:
                fp = os.path.join(args.real_images_root, f"{p[0]}.jpg")
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

def get_best_image_path(dir_path, safe_filename_prefix):
    # Final
    final = os.path.join(dir_path, f"{safe_filename_prefix}_FINAL.png")
    if os.path.exists(final): return final

    # Fallback V10..V1
    for i in range(10, 0, -1):
        p = os.path.join(dir_path, f"{safe_filename_prefix}_V{i}.png")
        if os.path.exists(p): return p

    return None

# --------------------------------------------------
# --- 4. Models ---
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
        import types
        mock_data = types.ModuleType("dinov3.data")
        mock_datasets = types.ModuleType("dinov3.data.datasets")
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

        models['dino_v3'] = torch.hub.load(args.dinov3_repo_path, 'dinov3_vitb16', source='local', pretrained=False)
        ckpt = torch.load(args.dinov3_weights_path, map_location='cpu')
        state_dict = ckpt.get('model', ckpt.get('teacher', ckpt))
        new_state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        models['dino_v3'].load_state_dict(new_state_dict, strict=False)
        models['dino_v3'] = models['dino_v3'].to(device).eval()
    except Exception as e:
        print(f"Error loading DINO v3: {e}")

    # 2. DINO v2/v1
    print("Loading DINO v2/v1...")
    models['dino_v2'] = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device).eval()
    models['dino_v1'] = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').to(device).eval()

    # 3. CLIP
    print("Loading CLIP...")
    models['clip_model'], _, models['clip_preprocess'] = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    models['clip_model'] = models['clip_model'].eval().to(device)
    models['clip_tokenizer'] = open_clip.get_tokenizer("ViT-B-32")

    # 4. SigLIP
    print("Loading SigLIP...")
    sig_name = 'hf-hub:timm/ViT-SO400M-14-SigLIP-384'
    models['siglip_model'], models['siglip_preprocess'] = create_model_from_pretrained(sig_name, device=device)
    models['siglip_model'] = models['siglip_model'].eval().to(device)
    models['siglip_tokenizer'] = get_tokenizer(sig_name)

    # 5. KID/FID (Shared instance for now, reset per experiment handled logic side)
    # Actually we need one instance per experiment to accumulate features?
    # No, we can process experiments sequentially for memory efficiency.
    print("Initializing Metric Calculators...")
    models['kid_transform'] = T.Compose([T.Resize(299), T.CenterCrop(299), T.ToTensor()])

    return models

def init_metrics_for_experiment(device):
    return {
        'kid': KernelInceptionDistance(subset_size=5, normalize=True).to(device),
        'fid': FrechetInceptionDistance(feature=2048, normalize=True).to(device),
        'is': InceptionScore(normalize=True).to(device)
    }

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

        # LapVar
        img_cv = cv2.imread(img_path)
        if img_cv is not None:
             gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
             scores['laplacian_var'] = cv2.Laplacian(gray, cv2.CV_64F).var()
        else: scores['laplacian_var'] = 0.0

        with torch.no_grad():
            # DINO v3
            dino_in = models['dino_transform'](img).unsqueeze(0).to(device)
            if 'dino_v3' in models:
                out = models['dino_v3'](dino_in)
                out = F.normalize(out, dim=-1)
                scores['dino_v3'] = F.cosine_similarity(out, real_feats_map['v3']).mean().item()

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

    except Exception:
        return None
    return scores

# --------------------------------------------------
# --- main ---
# --------------------------------------------------
def main():
    # 1. Discover Experiments
    print(f"Scanning for experiments in {args.root_dir}...")
    all_experiments = discover_experiments(args.root_dir)
    print(f"Found {len(all_experiments)} potential experiment folders.")

    # 2. Check Cache
    final_results = {}
    missing_experiments = []

    for exp_path in all_experiments:
        exp_name = os.path.relpath(exp_path, args.root_dir)
        log_file = os.path.join(exp_path, "logs", "evaluation_metrics.txt")

        if os.path.exists(log_file):
            metrics = parse_log_file(log_file)
            if metrics:
                final_results[exp_name] = metrics
                continue

        missing_experiments.append((exp_name, exp_path))

    print(f"\n{len(final_results)} experiments have cached results.")
    print(f"{len(missing_experiments)} experiments need evaluation.")

    # 3. Evaluate Missing
    if missing_experiments:
        # Load Models Once
        models = load_models(device)
        tasks = load_aircraft_tasks()

        # Iterate over missing experiments
        # Re-using the structure: Iterate tasks -> for each task, evaluate relevant exp
        # CAUTION: The original logic iterates TASKS and inside finds images for each method.
        # Here we have N experiments. We can group them by path?
        # Actually, it's efficient to verify if we can batch process or just loop TASKs once and check all MISSING experiments.

        # Initialize temp storage for missing experiments
        exp_accumulators = {exp_name: [] for exp_name, _ in missing_experiments}
        exp_metrics_instances = {exp_name: init_metrics_for_experiment(device) for exp_name, _ in missing_experiments}

        print("\nStarting evaluation loop...")
        for task in tqdm(tasks):
            # Precheck images for all missing exps
            active_exps = {}
            for exp_name, exp_path in missing_experiments:
                p = get_best_image_path(exp_path, task['safe_filename_prefix'])
                if p:
                    active_exps[exp_name] = p

            if not active_exps: continue # No images for this task in any missing exp

            # Load Real Data
            real_paths = task["real_image_paths"]
            if not real_paths: continue

            # Features
            real_feats = {}
            if 'dino_v3' in models:
                real_feats['v3'] = get_dino_features_batch(real_paths, models['dino_v3'], models['dino_transform'], device)
            else: real_feats['v3'] = None

            if any(v is None for k,v in real_feats.items() if k != 'v3'): continue

            # Text Features
            txt_feats = {}
            with torch.no_grad():
                ctk = models['clip_tokenizer']([task["prompt"]]).to(device)
                txt_feats['clip'] = models['clip_model'].encode_text(ctk)
                txt_feats['clip'] /= txt_feats['clip'].norm(dim=-1, keepdim=True)

                # SigLIP
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

            # Update Metrics (Real) for ALL active experiments
            # (Expensive? Reuse real features?)
            # KID/FID update needs features.
            use_paths = real_paths[:50]
            try:
                # Batch real images
                real_imgs = []
                for rp in use_paths:
                    real_imgs.append(models['kid_transform'](Image.open(rp).convert("RGB")))
                if real_imgs:
                    real_batch = torch.stack(real_imgs).to(device)
                    # Update all active experiments with REAL distribution
                    for exp_name in active_exps.keys():
                        exp_metrics_instances[exp_name]['kid'].update(real_batch, real=True)
                        exp_metrics_instances[exp_name]['fid'].update(real_batch, real=True)
            except Exception as e:
                # print(f"Error updating real metrics: {e}")
                pass

            # Evaluate Generated Images
            for exp_name, img_path in active_exps.items():
                # Single metrics
                res = evaluate_single(img_path, real_feats, txt_feats, models, device)
                if res:
                    exp_accumulators[exp_name].append(res)

                # Dist metrics (Fake)
                try:
                    fake_tensor = models['kid_transform'](Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                    exp_metrics_instances[exp_name]['kid'].update(fake_tensor, real=False)
                    exp_metrics_instances[exp_name]['is'].update(fake_tensor)
                    exp_metrics_instances[exp_name]['fid'].update(fake_tensor, real=False)
                except: pass

        # Compute & Save Results for New Evaluations
        for exp_name, exp_path in missing_experiments:
            acc = exp_accumulators[exp_name]
            if not acc: continue

            # Compute Final Scores
            m_res = {}
            m_res['clip'] = np.mean([x['clip'] for x in acc])
            m_res['siglip'] = np.mean([x['siglip'] for x in acc])
            m_res['dino_v3'] = np.mean([x.get('dino_v3', 0) for x in acc])
            m_res['lap_var'] = np.mean([x['laplacian_var'] for x in acc])

            # KID/FID
            try:
                kid_score, _ = exp_metrics_instances[exp_name]['kid'].compute()
                m_res['kid'] = kid_score.item()
            except: m_res['kid'] = 0.0

            try:
                m_res['fid'] = exp_metrics_instances[exp_name]['fid'].compute().item()
            except: m_res['fid'] = 0.0

            try:
                kl_mean, _ = exp_metrics_instances[exp_name]['is'].compute()
                m_res['inception_score'] = kl_mean.item()
            except: m_res['inception_score'] = 0.0

            # Add to final table
            final_results[exp_name] = m_res

            # Save to Logs
            try:
                log_dir = os.path.join(exp_path, "logs")
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, "evaluation_metrics.txt")
                with open(log_file, "w") as f:
                    f.write(f"Evaluation Metrics for {exp_name}\n")
                    f.write("="*50 + "\n")
                    f.write(f"CLIP Score: {m_res['clip']:.4f}\n")
                    f.write(f"SigLIP Score: {m_res['siglip']:.4f}\n")
                    f.write(f"DINO v3 Score: {m_res['dino_v3']:.4f}\n")
                    f.write(f"KID Score: {m_res['kid']:.5f}\n")
                    f.write(f"FID Score: {m_res['fid']:.4f}\n")
                    f.write(f"Laplacian Variance: {m_res['lap_var']:.1f}\n")
            except Exception as e:
                print(f"Failed to save log for {exp_name}: {e}")

    # 4. Final Big Table
    print("\n" + "="*150)
    print(f"{'Experiment Name':<60} | {'CLIP':<8} | {'SigLIP':<8} | {'DINOv3':<8} | {'KID':<8} | {'FID':<8} | {'IS':<8} | {'LapVar':<8}")
    print("-" * 150)

    # Sort keys for readability
    sorted_keys = sorted(final_results.keys())

    for name in sorted_keys:
        res = final_results[name]
        try:
            print(f"{name:<60} | {res.get('clip',0):.4f}   | {res.get('siglip',0):.4f}   | {res.get('dino_v3',0):.4f}   | {res.get('kid',0):.5f}  | {res.get('fid',0):.2f}    | {res.get('inception_score',0):.2f}    | {res.get('lap_var',0):.1f}")
        except: pass

    print("="*150)

    for name in sorted_keys:
        m = final_results[name]
        print(f"{name:<60} | {m.get('clip',0):.4f}   | {m.get('siglip',0):.4f}   | {m.get('dino_v3',0):.4f}   | {m.get('kid',0):.5f} | {m.get('fid',0):.4f} | {m.get('lap_var',0):.1f}")

    print("="*140)

if __name__ == "__main__":
    main()
