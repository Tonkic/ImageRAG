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
  - DINO v3 (Image Fidelity)
  - FID
  - Inception Score
'''

import os
import sys
import argparse
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
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from sklearn.covariance import LedoitWolf
from scipy import linalg

# --- Manual Metric Utils ---
def calculate_fid(act1, act2):
    # act1, act2: [N, 2048] numpy arrays
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean): covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_inception_score(logits, splits=1):
    # logits: [N, 1000] numpy
    # IS = exp( E_x [ KL( p(y|x) || p(y) ) ] )
    scores = []
    # softmax
    preds = F.softmax(torch.tensor(logits), dim=1).numpy()

    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

# --------------------------------------------------
# --- 1. Args ---
# --------------------------------------------------
parser = argparse.ArgumentParser(description="Recursive Evaluate All Results")
parser.add_argument("--device_id", type=str, default="0")
parser.add_argument("--root_dir", type=str, default="/home/tingyu/imageRAG/results")
# Use Aircraft dataset config by default as requested context seems to imply
parser.add_argument("--classes_txt", type=str, default="datasets/fgvc-aircraft-2013b/data/variants.txt")
parser.add_argument("--real_images_list", type=str, default="datasets/fgvc-aircraft-2013b/data/images_variant_test.txt")
parser.add_argument("--real_images_root", type=str, default="datasets/fgvc-aircraft-2013b/data/images")
parser.add_argument("--dinov3_repo_path", type=str, default="/home/tingyu/imageRAG/dinov3")
parser.add_argument("--dinov3_weights_path", type=str, default="/home/tingyu/imageRAG/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")

args = parser.parse_args()

# Multi-GPU Setup
if ',' in args.device_id:
    # E.g. "0,1" -> ex=cuda:0, met=cuda:1
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    ex_device = "cuda:0"
    met_device = "cuda:1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    ex_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    met_device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Using Extractors on: {ex_device}")
print(f"Using Metrics on: {met_device}")

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
                            elif "DINO v3" in key: metrics['dino_v3'] = val
                            elif "FID" in key: metrics['fid'] = val
                            elif "IS" in key or "Inception" in key: metrics['inception_score'] = val
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
def load_models(device, met_device=None):
    if met_device is None: met_device = device
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



    # 3. CLIP
    print("Loading CLIP...")
    models['clip_model'], _, models['clip_preprocess'] = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    models['clip_model'] = models['clip_model'].eval().to(device)
    models['clip_tokenizer'] = open_clip.get_tokenizer("ViT-B-32")

    print("Load InceptionV3 (Wrapped) for Manual FID/IS...")
    models['inception_transform'] = T.Compose([
        T.Resize(299, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        # Load standard inception v3
        inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        inception.fc = torch.nn.Identity() # Remove FC for features? No, FID uses pool3 (2048).
        # Inception V3 forward returns logits by default. We need internal Features.
        # We will use a Forward Hook or just return_nodes?
        # Standard InceptionV3 has 'aux_logits' which complicates things.
        # Let's use a simple wrapper class.
        class InceptionWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.model.eval()
                self.features = None

            def hook(self, module, input, output):
                self.features = output.flatten(1)

            def forward(self, x):
                # Register hook on 'avgpool' (which is effectively pool3 before fc)
                # In torchvision Inception3, it is named 'avgpool'.
                # output of avgpool is [N, 2048, 1, 1]
                h = self.model.avgpool.register_forward_hook(self.hook)
                logits = self.model(x)
                h.remove()
                return self.features, logits

        inception.eval().to(met_device)
        models['inception_model'] = InceptionWrapper(inception).to(met_device)
    except Exception as e:
        print(f"Error loading Inception: {e}")
        models['inception_model'] = None



    return models

def init_metrics_for_experiment(device):
    return {
        'fid': FrechetInceptionDistance(feature=2048, normalize=True).to(device),
        'is': InceptionScore(normalize=True).to(device)
    }

# --- CCMD & Stats Helpers ---
def compute_dataset_stats_from_map(real_map, models, device):
    print("Computing Dataset Statistics for CCMD...")
    # 1. Load all real images and embed with CLIP
    all_embeddings = []
    paths = []

    # Flatten map
    for cid, plist in real_map.items():
        for p in plist:
            all_embeddings.append(None) # Placeholder
            paths.append((cid, p))

    # Batch Process
    batch_size = 32
    embeddings_list = []
    labels_list = []
    path_list = []

    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i+batch_size]
            imgs = []
            valid_idx = []
            for k, (cid, p) in enumerate(batch):
                try:
                    img = Image.open(p).convert("RGB")
                    imgs.append(models['clip_preprocess'](img))
                    valid_idx.append(k)
                    labels_list.append(cid)
                    path_list.append(p)
                except: pass

            if not imgs: continue

            img_tensor = torch.stack(imgs).to(device)
            feats = models['clip_model'].encode_image(img_tensor)
            feats /= feats.norm(dim=-1, keepdim=True)
            embeddings_list.append(feats)

    if not embeddings_list: return None

    X = torch.cat(embeddings_list, dim=0) # [N, D]

    # Compute Means
    means = {}
    medoids = {}

    unique_labels = set(labels_list)
    X_centered_list = []
    embeddings_np = X.cpu().numpy()

    for lbl in unique_labels:
         # Indices for this class
         indices = [i for i, x in enumerate(labels_list) if x == lbl]
         cls_vecs = X[indices]

         # Mean
         mu = torch.mean(cls_vecs, dim=0)
         means[lbl] = mu

         # Medoid
         scores = (cls_vecs @ mu).cpu().numpy()
         best_idx = np.argmax(scores)
         medoids[lbl] = path_list[indices[best_idx]] # Store path

         # Center for Covariance
         cls_vecs_np = embeddings_np[indices]
         mu_np = np.mean(cls_vecs_np, axis=0)
         X_centered_list.append(cls_vecs_np - mu_np)

    # Covariance (Ledoit Wolf)
    print("Fitting Ledoit-Wolf Covariance...")
    X_centered = np.vstack(X_centered_list)
    cov = LedoitWolf(store_precision=True, assume_centered=True, block_size=1000)
    cov.fit(X_centered)
    precision = torch.tensor(cov.precision_, device=device, dtype=torch.float32)

    return {'means': means, 'precision': precision, 'medoids': medoids}

def get_ref_path_from_log(img_path):
    # Expects ../Boeing_707_V1.png
    # Log: ../Boeing_707.log
    # Content: ">> Selected Ref: ..." or ">> Static Ref: ..."
    try:
        dir_name = os.path.dirname(img_path)
        base = os.path.basename(img_path)
        # remove _V*.png or _FINAL.png
        if "_FINAL" in base: prefix = base.split("_FINAL")[0]
        elif "_V" in base: prefix = base.split("_V")[0]
        else: prefix = os.path.splitext(base)[0]

        log_path = os.path.join(dir_name, f"{prefix}.log")
        if not os.path.exists(log_path): return None

        with open(log_path, 'r') as f:
            for line in f:
                if ">> Selected Ref:" in line or ">> Static Ref:" in line:
                    # Parse path
                    # Format: ">> Selected Ref: datasets/fgvc.../123.jpg (Score..."
                    # Or ">> Static Ref: datasets/fgvc.../123.jpg (Score..."
                    if ">> Selected Ref:" in line:
                        parts = line.split(">> Selected Ref:", 1)[1].strip()
                    else:
                        parts = line.split(">> Static Ref:", 1)[1].strip()

                    # Remove score suffix if present
                    if " (" in parts: parts = parts.split(" (")[0]
                    return parts.strip()
    except: pass
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

def evaluate_single(img_path, class_name, real_feats_map, txt_feats, models, device):
    scores = {}
    try:
        img = Image.open(img_path).convert("RGB")

        # Shared DINO Features (Compute Once)
        dino_out_gen = None
        if 'dino_v3' in models:
             with torch.no_grad():
                 dino_in = models['dino_transform'](img).unsqueeze(0).to(device)
                 dino_out_gen = models['dino_v3'](dino_in)
                 dino_out_gen = F.normalize(dino_out_gen, dim=-1)
                 # Score against Real Set
                 scores['dino_v3'] = F.cosine_similarity(dino_out_gen, real_feats_map['v3']).mean().item()

        with torch.no_grad():
            # CLIP
            c_in = models['clip_preprocess'](img).unsqueeze(0).to(device)
            c_feat = models['clip_model'].encode_image(c_in)
            c_feat /= c_feat.norm(dim=-1, keepdim=True)
            scores['clip'] = (c_feat @ txt_feats['clip'].T).item()

            # --- New Metrics ---

            # 1. CCMD (Class-Conditional Mahalanobis Distance)
            if models.get('ccmd_stats') and class_name in models['ccmd_stats']['means']:
                 mu = models['ccmd_stats']['means'][class_name]
                 prec = models['ccmd_stats']['precision']

                 # Diff
                 diff = c_feat.squeeze(0) - mu
                 term1 = torch.mv(prec, diff)
                 dist_sq = torch.dot(diff, term1)
                 scores['ccmd'] = dist_sq.item()

            # 2. SVCG (Semantic-Visual Consistency Gap)
            if models.get('ccmd_stats') and class_name in models['ccmd_stats']['medoids']:
                 # Gap = | (1 - Sim(I, Proto)) - (1 - Sim(I, Text)) |
                 # Sim(I, Text) = scores['clip'] (Need to ensure txt_feats is correct class text)
                 # Wait, txt_feats is generic? No, evaluate_single called per image?
                 # Ah, evaluate_single is called in loop. txt_feats passed is dict 'clip': Tensor.
                 # Assuming txt_feats['clip'] is THE prompt for this image.

                 sim_text = scores['clip']

                 # Sim Visual Proto (Medoid)
                 medoid_path = models['ccmd_stats']['medoids'][class_name]
                 if os.path.exists(medoid_path):
                     # Load and embed Medoid (Should cache this?)
                     # For speed, we should cache medoid embeddings in stats.
                     # But current stats only return 'medoids' path dict.
                     # Let's re-encode for now (Optim: Cache later).
                     try:
                         m_img = Image.open(medoid_path).convert("RGB")
                         m_in = models['clip_preprocess'](m_img).unsqueeze(0).to(device)
                         m_feat = models['clip_model'].encode_image(m_in)
                         m_feat /= m_feat.norm(dim=-1, keepdim=True)
                         sim_vis = (c_feat @ m_feat.T).item()

                         gap = abs((1 - sim_vis) - (1 - sim_text))
                         scores['svcg'] = gap
                     except: pass

            # 3. RF (Reference Fidelity)
            ref_path = get_ref_path_from_log(img_path)
            if ref_path and os.path.exists(ref_path):
                 try:
                     # Load Reference and Compute DINO
                     r_img = Image.open(ref_path).convert("RGB")
                     r_in = models['dino_transform'](r_img).unsqueeze(0).to(device)

                     if 'dino_v3' in models and dino_out_gen is not None:
                         # Compute Ref DINO
                         r_out = models['dino_v3'](r_in)
                         r_out = F.normalize(r_out, dim=-1)

                         # Sim(Gen, Ref)
                         scores['rf'] = F.cosine_similarity(dino_out_gen, r_out).item()
                 except: pass

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
        models = load_models(ex_device, met_device)
        tasks = load_aircraft_tasks()

        # Compute Stats Mapping (Real Paths -> Class)
        print("Pre-computing stats for CCMD/SVCG...")
        real_map_by_class = {}
        # load_aircraft_tasks returns List[Dict]. Each dict has 'real_image_paths' and the class is embedded in prompt.
        # We need a simpler map: ClassName -> [paths].
        # Helper: Extract class name from prompt "a photo of a {name}"
        class_name_map = {}
        for t in tasks:
            cname = t['prompt'].replace("a photo of a ", "")
            real_map_by_class[cname] = t['real_image_paths']

        models['ccmd_stats'] = compute_dataset_stats_from_map(real_map_by_class, models, ex_device)

        # Iterate over missing experiments
        # Re-using the structure: Iterate tasks -> for each task, evaluate relevant exp
        # CAUTION: The original logic iterates TASKS and inside finds images for each method.
        # Here we have N experiments. We can group them by path?
        # Actually, it's efficient to verify if we can batch process or just loop TASKs once and check all MISSING experiments.

        # Initialize temp storage for missing experiments
        exp_accumulators = {exp_name: [] for exp_name, _ in missing_experiments}

        # [MEMORY FIX] Store features in CPU RAM, Compute Metrics LATER
        # real_features are computed cumulatively (or we can just store them)
        all_real_features = [] # List of [2048]
        # For each experiment: {'features': [], 'logits': []}
        exp_features_buffer = {exp_name: {'features': [], 'logits': []} for exp_name, _ in missing_experiments}

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
                real_feats['v3'] = get_dino_features_batch(real_paths, models['dino_v3'], models['dino_transform'], ex_device)
            else: real_feats['v3'] = None

            if any(v is None for k,v in real_feats.items() if k != 'v3'): continue

            # Text Features
            txt_feats = {}
            with torch.no_grad():
                ctk = models['clip_tokenizer']([task["prompt"]]).to(ex_device)
                txt_feats['clip'] = models['clip_model'].encode_text(ctk)
                txt_feats['clip'] /= txt_feats['clip'].norm(dim=-1, keepdim=True)

            # Update Metrics (Real) for ALL active experiments
            # (Expensive? Reuse real features?)
            # KID/FID update needs features.
            use_paths = real_paths[:50]
            try:
                # Batch real images
                real_imgs = []
                for rp in use_paths:
                    real_imgs.append(models['inception_transform'](Image.open(rp).convert("RGB")))
                if real_imgs:
                    real_batch = torch.stack(real_imgs).to(met_device)
                    # Extract Features for REAL
                    if models['inception_model']:
                        with torch.no_grad():
                            r_feats, _ = models['inception_model'](real_batch)
                            all_real_features.append(r_feats.cpu().numpy())
            except Exception as e:
                import traceback
                print(f"Error extracting REAL features: {e}\n{traceback.format_exc()}")

            # Evaluate Generated Images
            for exp_name, img_path in active_exps.items():
                # Single metrics
                # Extract class name from task prompt
                cname = task['prompt'].replace("a photo of a ", "")

                res = evaluate_single(img_path, cname, real_feats, txt_feats, models, ex_device)
                if res:
                    exp_accumulators[exp_name].append(res)

                # Dist metrics (Fake) buffer
                try:
                    fake_tensor = models['inception_transform'](Image.open(img_path).convert("RGB")).unsqueeze(0).to(met_device)
                    if models['inception_model']:
                        with torch.no_grad():
                            f_feats, f_logits = models['inception_model'](fake_tensor)
                            exp_features_buffer[exp_name]['features'].append(f_feats.cpu().numpy())
                            exp_features_buffer[exp_name]['logits'].append(f_logits.cpu().numpy())
                except Exception as e:
                     import traceback
                     print(f"Error extracting FAKE features for {exp_name}: {e}\n{traceback.format_exc()}")

        # Compute & Save Results for New Evaluations
        for exp_name, exp_path in missing_experiments:
            acc = exp_accumulators[exp_name]
            if not acc: continue

            # Compute Final Scores
            # Result Aggregation
            m_res = {}
            m_res['clip'] = np.mean([x['clip'] for x in acc])
            m_res['dino_v3'] = np.mean([x.get('dino_v3', 0) for x in acc])
            m_res['ccmd'] = np.mean([x.get('ccmd', 99.0) for x in acc])
            m_res['svcg'] = np.mean([x.get('svcg', 1.0) for x in acc]) # Default high gap
            m_res['rf'] = np.mean([x.get('rf', 0.0) for x in acc])

            # KID REMOVED

            # Compute IS/FID Manually
            try:
                if all_real_features and exp_features_buffer[exp_name]['features']:
                   # Concat
                   real_act = np.concatenate(all_real_features, axis=0) # [N_real, 2048]
                   fake_act = np.concatenate(exp_features_buffer[exp_name]['features'], axis=0)
                   fake_log = np.concatenate(exp_features_buffer[exp_name]['logits'], axis=0)

                   fid_val = calculate_fid(real_act, fake_act)
                   is_mean, is_std = calculate_inception_score(fake_log)

                   m_res['fid'] = fid_val
                   m_res['inception_score'] = is_mean
                else:
                   reason = "Empty feature buffer. "
                   if not all_real_features: reason += "all_real_features is empty. "
                   if not exp_features_buffer[exp_name]['features']: reason += "exp_features_buffer is empty. "
                   print(f"[{exp_name}] FID/IS skipped: {reason}")
                   try:
                       os.makedirs(os.path.join(exp_path, "logs"), exist_ok=True)
                       with open(os.path.join(exp_path, "logs", "error_log.txt"), "w") as ef:
                           ef.write(f"FID/IS computation skipped because: {reason}\n")
                   except: pass
                   m_res['fid'] = 999.0
                   m_res['inception_score'] = 0.0
            except Exception as ignored:
                print(f"[{exp_name}] FID/IS computation failed: {ignored}")
                import traceback
                try:
                    os.makedirs(os.path.join(exp_path, "logs"), exist_ok=True)
                    with open(os.path.join(exp_path, "logs", "error_log.txt"), "w") as ef:
                        ef.write(traceback.format_exc())
                except:
                    pass
                m_res['fid'] = 999.0
                m_res['inception_score'] = 0.0

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
                    f.write(f"DINO v3 Score: {m_res['dino_v3']:.4f}\n")
                    f.write(f"Reference Fidelity: {m_res['rf']:.4f}\n")
                    f.write(f"CCMD: {m_res['ccmd']:.4f}\n")
                    f.write(f"SVCG: {m_res['svcg']:.4f}\n")
                    f.write(f"FID Score: {m_res['fid']:.4f}\n")
                    f.write(f"Inception Score: {m_res['inception_score']:.4f}\n")
            except Exception as e:
                print(f"Failed to save log for {exp_name}: {e}")

    # 4. Final Big Table
    print("\n" + "="*150)
    print(f"{'Experiment Name':<60} | {'CLIP':<8} | {'RF':<8} | {'CCMD':<8} | {'SVCG':<8} | {'FID':<8} | {'IS':<8}")
    print("-" * 150)

    # Sort keys for readability
    sorted_keys = sorted(final_results.keys())

    for name in sorted_keys:
        res = final_results[name]
        try:
            print(f"{name:<60} | {res.get('clip',0):.4f}   | {res.get('rf',0):.4f}   | {res.get('ccmd',0):.2f}   | {res.get('svcg',0):.4f}   | {res.get('fid',0):.2f}    | {res.get('inception_score',0):.2f}")
        except: pass

    print("="*150)

    print("="*150)

if __name__ == "__main__":
    main()
