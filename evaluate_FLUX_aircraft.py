'''
Evaluation Script for Aircraft (FLUX Methods)
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
  - FID (Frechet Inception Distance)
  - Laplacian Variance (Blurriness)
  - Pairwise MS-SSIM (Diversity within generated group - if applicable)
'''

import os
import sys
import argparse
import cv2

# --------------------------------------------------
# --- 1. Args (Parse BEFORE importing torch) ---
# --------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate Aircraft (FLUX Methods)")
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
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure

# [New] Imports for ColQwen3
try:
    from transformers import AutoModel, AutoProcessor, AutoConfig
except ImportError:
    print("Transformers not found.")

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
    "Baseline": "results/FLUX_Baseline_Aircraft",
    "BC_SR": "results/FLUX_BC_SR_Aircraft",
    "BC_MGR": "results/FLUX_BC_MGR_Aircraft",
    "TAC_SR": "results/FLUX_TAC_SR_Aircraft",
    "TAC_MGR": "results/FLUX_TAC_MGR_Aircraft"
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

    # 6. KID & FID (One per method)
    print("Initializing KID & FID...")
    models['kid'] = {}
    models['fid'] = {}
    for m in METHODS.keys():
        # Reduced subset_size to 5 to allow calculation with fewer samples
        models['kid'][m] = KernelInceptionDistance(subset_size=5, normalize=True).to(device)
        models['fid'][m] = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    models['kid_transform'] = T.Compose([T.Resize(299), T.CenterCrop(299), T.ToTensor()])

    # 7. MS-SSIM
    print("Initializing MS-SSIM...")
    models['ms_ssim'] = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    models['ssim_transform'] = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    # 8. ColQwen3
    print("Loading ColQwen3...")
    try:
        # Load model directly
        models['colqwen3_model'] = AutoModel.from_pretrained(
            "TomoroAI/tomoro-colqwen3-embed-8b",
            trust_remote_code=True,
            dtype="auto"
        ).to(device).eval()
        models['colqwen3_processor'] = AutoProcessor.from_pretrained("TomoroAI/tomoro-colqwen3-embed-8b", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading ColQwen3: {e}")

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
        p = os.path.join(dir_path, f"{safe_filename_prefix}_V{i}.png")
        if os.path.exists(p):
            best_v_path = p
            break

    # If no V2-V10, check V1
    if not best_v_path:
        v1_p = os.path.join(dir_path, f"{safe_filename_prefix}_V1.png")
        if os.path.exists(v1_p):
            best_v_path = v1_p

    if best_v_path and os.path.exists(best_v_path):
        return best_v_path
    return None

def get_group_images(dir_path, safe_filename_prefix):
    """
    Find all group images for the latest retry.
    Pattern: {safe_filename_prefix}_retry{N}_group{M}.png
    """
    # Find max retry
    max_retry = -1
    pattern = os.path.join(dir_path, f"{safe_filename_prefix}_retry*_group*.png")
    files = glob.glob(pattern)

    if not files:
        return []

    for f in files:
        try:
            # Extract retry number
            base = os.path.basename(f)
            parts = base.split('_retry')
            if len(parts) > 1:
                retry_part = parts[1].split('_group')[0]
                retry_num = int(retry_part)
                if retry_num > max_retry:
                    max_retry = retry_num
        except: pass

    if max_retry == -1:
        return []

    # Collect all images for max_retry
    group_images = []
    target_pattern = os.path.join(dir_path, f"{safe_filename_prefix}_retry{max_retry}_group*.png")
    for f in glob.glob(target_pattern):
        group_images.append(f)

    return group_images

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

def calculate_pairwise_ms_ssim(img_paths, model, transform, device):
    if len(img_paths) < 2:
        return None
    
    # Randomly pick 2
    pair = random.sample(img_paths, 2)
    
    try:
        img1 = transform(Image.open(pair[0]).convert("RGB")).unsqueeze(0).to(device)
        img2 = transform(Image.open(pair[1]).convert("RGB")).unsqueeze(0).to(device)
        
        with torch.no_grad():
            score = model(img1, img2)
        return score.item()
    except Exception as e:
        # print(f"Error calculating MS-SSIM: {e}")
        return None

def evaluate_single(img_path, real_feats_map, txt_feats, models, device, prompt=None):
    scores = {}
    try:
        img = Image.open(img_path).convert("RGB")
        
        # Laplacian Variance
        scores['laplacian_var'] = calculate_laplacian_variance(img_path)

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

            # ColQwen3 MaxSim
            if 'colqwen3' in txt_feats and 'colqwen3_model' in models:
                processor = models['colqwen3_processor']
                model = models['colqwen3_model']

                # Process Image
                if hasattr(processor, "process_images"):
                    batch_images = processor.process_images([img]).to(device)
                    image_out = model(**batch_images)
                    if hasattr(image_out, 'embeddings'):
                        image_emb = image_out.embeddings
                    elif hasattr(image_out, 'last_hidden_state'):
                        image_emb = image_out.last_hidden_state
                    else:
                        image_emb = image_out
                else:
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    image_out = model(**inputs)
                    if hasattr(image_out, 'embeddings'):
                        image_emb = image_out.embeddings
                    elif hasattr(image_out, 'last_hidden_state'):
                        image_emb = image_out.last_hidden_state
                    else:
                        image_emb = image_out

                # MaxSim
                Q = txt_feats['colqwen3'].to(device)
                D = image_emb.to(device)

                # Ensure dtype match
                D = D.to(Q.dtype)

                sim_matrix = torch.einsum('bqd,bnd->bqn', Q, D)
                max_sim = sim_matrix.max(dim=-1).values
                score = max_sim.sum(dim=-1).item()
                scores['colqwen3_maxsim'] = score

    except Exception as e:
        # print(f"Error evaluating {img_path}: {e}")
        return None
    return scores

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

    # 2. Pre-scan Counts
    print(f"\nScanning generated images for {len(tasks)} tasks...")
    counts = {m: 0 for m in METHODS.keys()}
    for task in tasks:
        for method_name, dir_path in METHODS.items():
            if get_best_image_path(method_name, dir_path, task['safe_filename_prefix']):
                counts[method_name] += 1

    print(f"  Evaluated Images Count (Total Tasks: {len(tasks)}):")
    for m in METHODS.keys():
        print(f"    - {m:<15}: {counts[m]}")
    print("-" * 80)

    models = load_models(device)

    # Storage for results
    results = {m: [] for m in METHODS.keys()}

    print(f"\nStarting evaluation on {len(tasks)} Aircraft classes...")

    for task in tqdm(tasks):
        # [Optimization] Pre-check if any generated image exists
        task_img_paths = {}
        for method_name, dir_path in METHODS.items():
            p = get_best_image_path(method_name, dir_path, task['safe_filename_prefix'])
            if p:
                task_img_paths[method_name] = p

        if not task_img_paths:
            continue

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

            # SigLIP Tokenizer Fix
            try:
                stk = models['siglip_tokenizer']([task["prompt"]]).to(device)
            except AttributeError:
                # Fallback for T5Tokenizer issues in some transformers versions
                if hasattr(models['siglip_tokenizer'], 'tokenizer'):
                    # It's the HFTokenizer wrapper
                    internal_tok = models['siglip_tokenizer'].tokenizer
                    if hasattr(internal_tok, '__call__'):
                        res = internal_tok(
                            [task["prompt"]],
                            padding='max_length',
                            truncation=True,
                            max_length=64,
                            return_tensors='pt'
                        )
                        stk = res['input_ids'].to(device)
                    else:
                        raise
                else:
                    raise

            txt_feats['siglip'] = models['siglip_model'].encode_text(stk)
            txt_feats['siglip'] = F.normalize(txt_feats['siglip'], dim=-1)

            # ColQwen3
            if 'colqwen3_model' in models:
                try:
                    processor = models['colqwen3_processor']
                    model = models['colqwen3_model']
                    with torch.no_grad():
                        if hasattr(processor, "process_texts"):
                            batch_queries = processor.process_texts([task["prompt"]])
                            batch_queries = {k: v.to(device) for k, v in batch_queries.items()}
                            query_outputs = model(**batch_queries)
                            txt_feats['colqwen3'] = query_outputs.embeddings if hasattr(query_outputs, 'embeddings') else query_outputs
                        else:
                            inputs = processor(text=[task["prompt"]], return_tensors="pt", padding=True).to(device)
                            out = model(**inputs)
                            txt_feats['colqwen3'] = out.embeddings if hasattr(out, 'embeddings') else out.last_hidden_state
                except Exception as e:
                    print(f"Error encoding text for ColQwen3: {e}")

        # 3. Update KID & FID (Real) - Update ALL KID/FID models with real data
        kp = random.sample(real_paths, min(len(real_paths), 20))
        kimgs = []
        for p in kp:
            try: kimgs.append(models['kid_transform'](Image.open(p).convert("RGB")))
            except: pass

        if kimgs:
            kt = torch.stack(kimgs).to(device)
            
            for m in METHODS.keys():
                try:
                    models['kid'][m].update(kt, real=True)
                    models['fid'][m].update(kt, real=True)
                except Exception as e:
                    print(f"Warning: KID/FID update failed for {m} (real): {e}")

        # 4. Evaluate Each Method
        for method_name, dir_path in METHODS.items():
            img_path = task_img_paths.get(method_name)

            if not img_path:
                continue

            # Eval Single Image Metrics
            res = evaluate_single(img_path, real_feats, txt_feats, models, device, prompt=task["prompt"])
            
            # Eval Pairwise MS-SSIM (Diversity)
            # Find group images
            group_imgs = get_group_images(dir_path, task['safe_filename_prefix'])
            ms_ssim_val = calculate_pairwise_ms_ssim(group_imgs, models['ms_ssim'], models['ssim_transform'], device)
            if ms_ssim_val is not None:
                res['ms_ssim'] = ms_ssim_val

            if res:
                results[method_name].append(res)
                try:
                    img_tensor = models['kid_transform'](Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                    models['kid'][method_name].update(img_tensor, real=False)
                    models['fid'][method_name].update(img_tensor, real=False)
                except Exception as e:
                    # print(f"Warning: KID/FID update failed for {method_name} (fake): {e}")
                    pass

    # Report
    print("\n" + "="*120)
    print(f"  EVAL REPORT: Aircraft (FLUX Methods)")
    print("="*120)

    print(f"{'Method':<15} | {'CLIP':<8} | {'SigLIP':<8} | {'DINOv3':<8} | {'KID':<8} | {'FID':<8} | {'LapVar':<8} | {'MS-SSIM':<8} | {'MaxSim':<8}")
    print("-" * 120)

    for method_name in METHODS.keys():
        lst = results[method_name]
        if not lst:
            print(f"{method_name:<15} | N/A")
            continue

        # Compute KID & FID
        try:
            kid_score, _ = models['kid'][method_name].compute()
            kid_val = kid_score.item()
        except: kid_val = 0.0
        
        try:
            fid_val = models['fid'][method_name].compute().item()
        except: fid_val = 0.0

        clip_s = np.mean([x['clip'] for x in lst])
        siglip_s = np.mean([x['siglip'] for x in lst])
        d3 = np.mean([x['dinov3_base'] for x in lst])
        cq3 = np.mean([x.get('colqwen3_maxsim', 0) for x in lst])
        lap_var = np.mean([x.get('laplacian_var', 0) for x in lst])
        
        # MS-SSIM (Filter None)
        ssim_vals = [x.get('ms_ssim') for x in lst if x.get('ms_ssim') is not None]
        ms_ssim = np.mean(ssim_vals) if ssim_vals else 0.0

        print(f"{method_name:<15} | {clip_s:.4f}   | {siglip_s:.4f}   | {d3:.4f}   | {kid_val:.5f} | {fid_val:.4f} | {lap_var:.1f}    | {ms_ssim:.4f}   | {cq3:.4f}")

    print("="*120)

if __name__ == "__main__":
    main()
