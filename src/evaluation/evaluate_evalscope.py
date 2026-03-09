import os
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm
import gc
import torch
import pandas as pd
import multiprocessing
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy import linalg
import sys
import datetime
import traceback

class TeeLogger(object):
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_tee_logging(log_filename):
    if not hasattr(sys.stdout, "log"):
        sys.stdout = TeeLogger(log_filename, sys.stdout)
        sys.stderr = TeeLogger(log_filename, sys.stderr)

# Metric Descriptions (Higher is better unless noted)
METRIC_INFO = {
    "CLIPScore": "Measures image-text alignment using CLIP ViT-L/14.",
    "ImageReward": "Human preference reward model based on human feedback.",
    "PickScore": "Preference model trained on Pick-a-Pic dataset.",
    "VQAScore": "Visual Question Answering accuracy for prompt adherence.",
    "BLIPv2Score": "Image-text matching score using BLIP-2.",
    "HPSv2Score": "Human Preference Score v2 trained on HPD v2.",
    "HPSv2.1Score": "Updated Human Preference Score v2.1.",
    "MPS": "Multi-modal Preference Score.",
    "FGA_BLIP2Score": "Fine-Grained Alignment score using BLIP-2.",
    "FID Score": "Fréchet Inception Distance. Measures distribution distance. (LOWER is Better)",
    "Inception Score": "Measures diversity and quality.",
    "DINO v3 Score": "Image similarity using DINOv3 features."
}

try:
    try:
        from evalscope.metrics import (
            CLIPScoreMetric, ImageRewardMetric, VQAScore, PickScore,
            BLIPv2Score, HPSv2Score, HPSv2_1Score, MPS, FGA_BLIP2Score
        )
    except ImportError:
        try:
            from evalscope.metrics.vl import (
                CLIPScoreMetric, ImageRewardMetric, VQAScore, PickScore,
                BLIPv2Score, HPSv2Score, HPSv2_1Score, MPS, FGA_BLIP2Score
            )
        except ImportError:
            from evalscope.metrics.metric import (
                CLIPScore, ImageRewardScore, VQAScore, PickScore,
                BLIPv2Score, HPSv2Score, HPSv2_1Score, MPS, FGA_BLIP2Score
            )
            CLIPScoreMetric = CLIPScore
            ImageRewardMetric = ImageRewardScore
    EVALSCOPE_AVAILABLE = True
except ImportError as e:
    EVALSCOPE_AVAILABLE = False
    print(f"Warning: EvalScope vl metrics could not be imported. Error: {e}")

def is_experiment_folder(path, allow_incomplete=False):
    if not os.path.isdir(path): return False

    # Collect all unique class names generated in this folder
    generated_classes = set()

    # Check FINALs
    for p in glob.glob(os.path.join(path, "*_FINAL.png")):
        base = os.path.basename(p)
        class_name = base.replace("_FINAL.png", "")
        generated_classes.add(class_name)

    # Check V versions
    for p in glob.glob(os.path.join(path, "*_V*.png")):
        base = os.path.basename(p)
        # Extract class name before _V
        class_name = base.rsplit("_V", 1)[0]
        generated_classes.add(class_name)

    if len(generated_classes) >= 100:
        return True
    elif len(generated_classes) > 0:
        if allow_incomplete:
            print(f"[Include] Experiment '{os.path.basename(path)}' only has {len(generated_classes)}/100 classes, but --allow_incomplete is set.")
            return True
        else:
            print(f"[Skip] Experiment '{os.path.basename(path)}' only has {len(generated_classes)}/100 classes. Evaluator will skip until 100 classes are completed.")
            return False

    return False

def discover_experiments(root, allow_incomplete=False):
    experiments = []
    for root_path, dirs, files in os.walk(root):
        if is_experiment_folder(root_path, allow_incomplete):
            experiments.append(root_path)
    return experiments

def get_best_image_path(dir_path, safe_filename_prefix):
    final = os.path.join(dir_path, f"{safe_filename_prefix}_FINAL.png")
    if os.path.exists(final): return final
    max_v = -1
    best_p = None
    for p in glob.glob(os.path.join(dir_path, f"{safe_filename_prefix}_V*.png")):
        try:
            v_num = int(os.path.basename(p).split('_V')[-1].split('.png')[0])
            if v_num > max_v:
                max_v = v_num
                best_p = p
        except: pass
    return best_p

def load_metric_evaluator(metric_name):
    try:
        print(f"Loading {metric_name} model to GPU...")
        if metric_name == "CLIPScore": return CLIPScoreMetric()
        elif metric_name == "ImageReward": return ImageRewardMetric()
        elif metric_name == "PickScore": return PickScore()
        elif metric_name == "VQAScore": return VQAScore()
        elif metric_name == "BLIPv2Score": return BLIPv2Score()
        elif metric_name == "HPSv2Score": return HPSv2Score()
        elif metric_name == "HPSv2.1Score": return HPSv2_1Score()
        elif metric_name == "MPS": return MPS()
        elif metric_name == "FGA_BLIP2Score":
            torch.cuda.empty_cache()
            return FGA_BLIP2Score()
        else: return None
    except Exception as e:
        print(f"Failed to load {metric_name}: {e}")
        if "CUDA out of memory" in str(e):
             torch.cuda.empty_cache()
             gc.collect()
        return None

def run_single_metric(metric_name, root_dir, classes_txt, device_id, force_rerun=False, log_file=None):
    if log_file: setup_tee_logging(log_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    experiments = discover_experiments(root_dir)

    exps_to_process = []
    for exp_path in experiments:
        log_file = os.path.join(exp_path, "logs", "evalscope_metrics.json")
        needs_run = True
        if not force_rerun and os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    if "summary" in data and metric_name in data["summary"]:
                         needs_run = False
            except: pass
        if needs_run: exps_to_process.append(exp_path)

    if not exps_to_process:
        print(f"[{metric_name}] All experiments cached. Skipping.")
        return

    if not os.path.exists(classes_txt): return
    with open(classes_txt, 'r') as f: classes = [l.strip() for l in f.readlines() if l.strip()]

    evaluator = load_metric_evaluator(metric_name)
    if evaluator is None: return

    for exp_path in tqdm(exps_to_process, desc=f"Evaluating {metric_name}"):
        try:
            eval_data = []
            for cls in classes:
                safe_name = cls.replace(" ", "_").replace("/", "-")
                img_path = get_best_image_path(exp_path, safe_name)
                if img_path: eval_data.append({"class": cls, "prompt": f"a photo of a {cls}", "image": img_path})
            if not eval_data: continue

            log_dir = os.path.join(exp_path, "logs")
            log_file = os.path.join(log_dir, "evalscope_metrics.json")
            current_data = {"summary": {}, "details": []}

            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f: current_data = json.load(f)
                except: pass

            details_map = {d.get("class"): d for d in current_data.get("details", [])}
            scores = []
            for item in eval_data:
                try:
                    res = evaluator(image=item['image'], text=item['prompt'])
                    val = list(res.values())[0] if isinstance(res, dict) else (res[0] if isinstance(res, (list, tuple)) else res)
                    val = float(val)
                    scores.append(val)
                    c = item["class"]
                    if c not in details_map: details_map[c] = {"class": c}
                    details_map[c][metric_name] = val
                except: pass

            if scores:
                if "summary" not in current_data: current_data["summary"] = {}
                current_data["summary"][metric_name] = float(np.mean(scores))
                current_data["details"] = [details_map[cls] for cls in classes if cls in details_map]
                os.makedirs(log_dir, exist_ok=True)
                with open(log_file, "w") as f: json.dump(current_data, f, indent=4)
        except Exception as e:
            print(f"Error processing {exp_path}: {e}")
            traceback.print_exc()

    del evaluator
    gc.collect()
    torch.cuda.empty_cache()

# ================================
# Manual Metric Utils (IS, FID)
# ================================
def calculate_fid(act1, act2):
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean): covmean = covmean.real
    return float(ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean))

def calculate_inception_score(logits, splits=1):
    scores = []
    preds = F.softmax(torch.tensor(logits), dim=1).numpy()
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return float(np.mean(scores))

class InceptionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.features = None
    def hook(self, module, input, output):
        self.features = output.flatten(1)
    def forward(self, x):
        h = self.model.avgpool.register_forward_hook(self.hook)
        logits = self.model(x)
        h.remove()
        return self.features, logits

def run_fid_is_metric(root_dir, classes_txt, real_images_list, real_images_root, device_id, force_rerun=False, log_file=None):
    if log_file: setup_tee_logging(log_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiments = discover_experiments(root_dir)

    exps_to_process = []
    for exp_path in experiments:
        log_file = os.path.join(exp_path, "logs", "evalscope_metrics.json")
        needs_run = True
        if not force_rerun and os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    if "summary" in data and "FID Score" in data["summary"] and "Inception Score" in data["summary"]:
                         needs_run = False
            except: pass
        if needs_run: exps_to_process.append(exp_path)

    if not exps_to_process:
        print("[FID/IS] All experiments cached. Skipping.")
        return

    # Load Real Images Map
    class_map = {}
    with open(classes_txt) as f:
        for i, l in enumerate(f):
            if l.strip(): class_map[l.strip()] = i

    real_paths = []
    with open(real_images_list) as f:
        for l in f:
            p = l.strip().split(' ', 1)
            if len(p) == 2 and p[1] in class_map:
                fp = os.path.join(real_images_root, f"{p[0]}.jpg")
                if os.path.exists(fp): real_paths.append(fp)

    if not real_paths:
        print("[FID/IS] No real images found. Skipping.")
        return

    print("[FID/IS] Loading InceptionV3 to GPU...")
    inception_transform = T.Compose([
        T.Resize(299, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        import sys
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        inception.fc = torch.nn.Identity()
        sys.stdout.close()
        sys.stdout = old_stdout
        inception_model = InceptionWrapper(inception).to(device)
    except Exception as e:
        print(f"[FID/IS] Error loading InceptionV3: {e}")
        return

    print("[FID/IS] Extracting real features...")
    all_real_features = []
    for i in range(0, len(real_paths), 64):
        batch = real_paths[i:i+64]
        imgs = []
        for p in batch:
            try: imgs.append(inception_transform(Image.open(p).convert("RGB")))
            except: pass
        if imgs:
            with torch.no_grad():
                f_feats, _ = inception_model(torch.stack(imgs).to(device))
                all_real_features.append(f_feats.cpu().numpy())
    if not all_real_features: return
    real_act = np.concatenate(all_real_features, axis=0)
    if len(real_act) < 2:
        print("[FID/IS] Warning: Not enough real images to compute statistics (requires >=2). Skipping.")
        return

    for exp_path in tqdm(exps_to_process, desc="Evaluating FID & IS"):
        exp_feats, exp_logits = [], []
        for cls in class_map.keys():
            safe_name = cls.replace(" ", "_").replace("/", "-")
            img_path = get_best_image_path(exp_path, safe_name)
            if img_path:
                try:
                    img = inception_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                    with torch.no_grad():
                        f_feats, f_log = inception_model(img)
                        exp_feats.append(f_feats.cpu().numpy())
                        exp_logits.append(f_log.cpu().numpy())
                except: pass

        if exp_feats:
            fake_act = np.concatenate(exp_feats, axis=0)
            fake_log = np.concatenate(exp_logits, axis=0)

            if len(fake_act) < 2:
                print(f"\n[FID/IS] Warning: Found only {len(fake_act)} image(s) in {exp_path}. FID computation requires at least 2 samples. Skipping this exp.")
                continue
            fid_val = calculate_fid(real_act, fake_act)
            is_val = calculate_inception_score(fake_log)

            log_dir = os.path.join(exp_path, "logs")
            log_file = os.path.join(log_dir, "evalscope_metrics.json")
            current_data = {"summary": {}, "details": []}
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f: current_data = json.load(f)
                except: pass
            if "summary" not in current_data: current_data["summary"] = {}
            current_data["summary"]["FID Score"] = fid_val
            current_data["summary"]["Inception Score"] = is_val

            os.makedirs(log_dir, exist_ok=True)
            with open(log_file, "w") as f: json.dump(current_data, f, indent=4)

    del inception_model
    gc.collect()
    torch.cuda.empty_cache()


def run_dino_metric(root_dir, classes_txt, real_images_list, real_images_root, repo_path, weights_path, device_id, force_rerun=False, log_file=None):
    if log_file: setup_tee_logging(log_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import sys

    experiments = discover_experiments(root_dir)
    exps_to_process = []
    for exp_path in experiments:
        log_file = os.path.join(exp_path, "logs", "evalscope_metrics.json")
        needs_run = True
        if not force_rerun and os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    if "summary" in data and "DINO v3 Score" in data["summary"]: needs_run = False
            except: pass
        if needs_run: exps_to_process.append(exp_path)

    if not exps_to_process:
        print("[DINOv3] All experiments cached. Skipping.")
        return

    class_map = {}
    with open(classes_txt) as f:
        for i, l in enumerate(f):
            if l.strip(): class_map[l.strip()] = i

    real_map = {c: [] for c in class_map.keys()}
    with open(real_images_list) as f:
        for l in f:
            p = l.strip().split(' ', 1)
            if len(p) == 2 and p[1] in class_map:
                fp = os.path.join(real_images_root, f"{p[0]}.jpg")
                if os.path.exists(fp): real_map[p[1]].append(fp)

    print("[DINOv3] Loading DINO v3 to GPU...")
    dino_transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    sys.path.append(os.path.abspath(repo_path))
    try:
        import types
        mock_data = types.ModuleType("dinov3.data")
        mock_datasets = types.ModuleType("dinov3.data.datasets")
        mock_data.DatasetWithEnumeratedTargets = type("DatasetWithEnumeratedTargets", (), {})
        mock_data.SamplerType = type("SamplerType", (), {})
        mock_data.ImageDataAugmentation = type("ImageDataAugmentation", (), {})
        mock_data.make_data_loader = lambda *a, **k: None
        mock_data.datasets = mock_datasets
        sys.modules["dinov3.data"] = mock_data
        sys.modules["dinov3.data.datasets"] = mock_datasets

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dino_v3 = torch.hub.load(repo_path, 'dinov3_vitb16', source='local', pretrained=False)
            ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
            state_dict = ckpt.get('model', ckpt.get('teacher', ckpt))
            new_state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            dino_v3.load_state_dict(new_state_dict, strict=False)
            dino_v3 = dino_v3.to(device).eval()
    except Exception as e:
        print(f"[DINOv3] Error loading model: {e}")
        return

    print("[DINOv3] Extracting real features...")
    real_feats = {}
    for cname, paths in real_map.items():
        if not paths: continue
        imgs = []
        for p in paths[:50]:
            try: imgs.append(dino_transform(Image.open(p).convert("RGB")))
            except: pass
        if imgs:
            with torch.no_grad():
                f = dino_v3(torch.stack(imgs).to(device))
                f = F.normalize(f, dim=-1)
                real_feats[cname] = f

    for exp_path in tqdm(exps_to_process, desc="Evaluating DINOv3"):
        log_dir = os.path.join(exp_path, "logs")
        log_file = os.path.join(log_dir, "evalscope_metrics.json")
        current_data = {"summary": {}, "details": []}
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f: current_data = json.load(f)
            except: pass

        details_map = {d.get("class"): d for d in current_data.get("details", [])}
        scores = []

        for cls in class_map.keys():
            if cls not in real_feats: continue
            safe_name = cls.replace(" ", "_").replace("/", "-")
            img_path = get_best_image_path(exp_path, safe_name)
            if img_path:
                try:
                    img = dino_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = dino_v3(img)
                        out = F.normalize(out, dim=-1)
                        score = float(F.cosine_similarity(out, real_feats[cls]).mean().item())
                        scores.append(score)
                        if cls not in details_map: details_map[cls] = {"class": cls}
                        details_map[cls]["DINO v3 Score"] = score
                except: pass

        if scores:
            if "summary" not in current_data: current_data["summary"] = {}
            current_data["summary"]["DINO v3 Score"] = float(np.mean(scores))
            current_data["details"] = [details_map[cls] for cls in class_map.keys() if cls in details_map]

            os.makedirs(log_dir, exist_ok=True)
            with open(log_file, "w") as f: json.dump(current_data, f, indent=4)

    del dino_v3
    gc.collect()
    torch.cuda.empty_cache()


def main():
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Evaluate T2I with EvalScope Metrics (Memory Optimized)")
    parser.add_argument("--root_dir", type=str, default="/home/tingyu/imageRAG/results")
    parser.add_argument("--classes_txt", type=str, default="datasets/fgvc-aircraft-2013b/data/variants.txt")
    parser.add_argument("--device_id", type=str, default="0", help="GPU device ID")
    parser.add_argument("--metrics", type=str, default="all", help="Comma-sep list or 'all'")
    parser.add_argument("--real_images_list", type=str, default="datasets/fgvc-aircraft-2013b/data/images_variant_test.txt")
    parser.add_argument("--real_images_root", type=str, default="datasets/fgvc-aircraft-2013b/data/images", help="For FID/IS")
    parser.add_argument("--dinov3_repo_path", type=str, default="/home/tingyu/imageRAG/dinov3")
    parser.add_argument("--dinov3_weights_path", type=str, default="/home/tingyu/imageRAG/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation even if cached")
    parser.add_argument("--update_config_only", action="store_true", help="Only update config columns in CSV without running any evaluations")
    parser.add_argument("--allow_incomplete", action="store_true", help="Evaluate experiments even if they have < 100 classes")
    args = parser.parse_args()

    os.makedirs(args.root_dir, exist_ok=True)
    log_filename = os.path.join(args.root_dir, f"evalscope_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    setup_tee_logging(log_filename)
    print(f"[INFO] Logging all stdout and stderr to: {log_filename}")

    print(f"Scanning for experiments in {args.root_dir}...")
    all_experiments = discover_experiments(args.root_dir, args.allow_incomplete)
    print(f"Found {len(all_experiments)} potential experiment folders.")

    metric_names = [m.strip() for m in args.metrics.split(",")]
    if "all" in metric_names:
        metric_names = [
            "CLIPScore", "ImageReward", "PickScore",
            "VQAScore", "BLIPv2Score", "HPSv2Score",
            "HPSv2.1Score", "MPS", "FGA_BLIP2Score"
        ]

    try:
        if not args.update_config_only:
            for m_name in metric_names:
                if m_name in ["FID Score", "Inception Score", "DINO v3 Score"]: continue
                print(f"\n" + "="*60)
                print(f" STARTING EVALUATION FOR: {m_name}")
                print("="*60)
                p = multiprocessing.Process(
                    target=run_single_metric,
                    args=(m_name, args.root_dir, args.classes_txt, args.device_id, args.force, log_filename)
                )
                p.start()
                while p.is_alive():
                    p.join(timeout=1.0)

            # DINO v3 Process
            if "all" in args.metrics or "DINO v3 Score" in args.metrics:
                print(f"\n" + "="*60)
                print(f" STARTING EVALUATION FOR: DINO v3 Score")
                print("="*60)
                p = multiprocessing.Process(
                    target=run_dino_metric,
                    args=(args.root_dir, args.classes_txt, args.real_images_list, args.real_images_root,
                          args.dinov3_repo_path, args.dinov3_weights_path, args.device_id, args.force, log_filename)
                )
                p.start()
                while p.is_alive():
                    p.join(timeout=1.0)

            # FID and IS Process
            if "all" in args.metrics or "FID Score" in args.metrics or "Inception Score" in args.metrics:
                print(f"\n" + "="*60)
                print(f" STARTING EVALUATION FOR: FID & Inception Score")
                print("="*60)
                p = multiprocessing.Process(
                    target=run_fid_is_metric,
                    args=(args.root_dir, args.classes_txt, args.real_images_list, args.real_images_root,
                          args.device_id, args.force, log_filename)
                )
                p.start()
                while p.is_alive():
                    p.join(timeout=1.0)
        else:
            print("\n[INFO] Skipping actual metrics evaluation (--update_config_only is ON).")

        print("\nEvaluation Complete.")
        generate_summary_table(args.root_dir)

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected. Terminating child processes...")
        if 'p' in locals() and p.is_alive():
            p.terminate()
            p.join()
        print("[!] Evaluation aborted by user.")
        sys.exit(130)


def extract_run_config(exp_path):
    """Parse logs/run_config.txt to extract key experimental hyperparameters."""
    config_file = os.path.join(exp_path, "logs", "run_config.txt")
    config_data = {}
    if not os.path.exists(config_file):
        return config_data

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Parse Command Line Arguments
                if line.startswith("retrieval_method:"):
                    config_data["Retrieval_Method"] = line.split(":", 1)[1].strip()
                elif line.startswith("var_k:"):
                    config_data["VAR_K"] = line.split(":", 1)[1].strip()
                elif line.startswith("retrieval_datasets:"):
                    # Example: "--retrieval_datasets aircraft cub imagenet"
                    # But if it's stored as "retrieval_datasets: ['aircraft', 'cub']", handle string rep:
                    val = line.split(":", 1)[1].strip()
                    config_data["Retrieval_Datasets"] = val.replace("[", "").replace("]", "").replace("'", "")
                elif line.startswith("llm_model:"):
                    config_data["LLM_Model"] = line.split(":", 1)[1].strip()
                elif line.startswith("image_guidance_scale:"):
                    config_data["Image_Guidance"] = line.split(":", 1)[1].strip()
                elif line.startswith("text_guidance_scale:"):
                    config_data["Text_Guidance"] = line.split(":", 1)[1].strip()
                elif line.startswith("dino_lambda_init:"):
                    config_data["DINO_L_Init"] = line.split(":", 1)[1].strip()
                elif line.startswith("dino_lambda_max:"):
                    config_data["DINO_L_Max"] = line.split(":", 1)[1].strip()
                elif line.startswith("tac_pass_threshold:"):
                    config_data["TAC_Pass"] = line.split(":", 1)[1].strip()
                elif line.startswith("tac_early_stop_threshold:"):
                    config_data["TAC_EarlyStop"] = line.split(":", 1)[1].strip()
                elif line.startswith("height:") or line.startswith("width:"):
                    # Quick string concat for resolution
                    val = line.split(":", 1)[1].strip()
                    if "Resolution" in config_data:
                        config_data["Resolution"] = f"{config_data['Resolution']}x{val}"
                    else:
                        config_data["Resolution"] = val
    except Exception as e:
        print(f"  [Warning] Failed to parse config for {exp_path}: {e}")

    return config_data


def generate_summary_table(root_dir):
    experiments = discover_experiments(root_dir)
    results = {}

    for exp_path in experiments:
        exp_name = os.path.relpath(exp_path, root_dir)
        log_file = os.path.join(exp_path, "logs", "evalscope_metrics.json")
        exp_results = {}

        # 1. Read Metrics
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    summary = json.load(f).get("summary", {})
                    if summary:
                        summary.pop("Laplacian Variance", None)
                        summary.pop("SigLIP Score", None)
                        exp_results.update(summary)
            except: pass

        # 2. Read Config
        config_data = extract_run_config(exp_path)
        if config_data:
            exp_results.update(config_data)

        if exp_results:
            results[exp_name] = exp_results

    gt_file_candidates = [
        os.path.join(root_dir, "groundtruth_metrics.txt"),
        os.path.join(root_dir, "..", "groundtruth_metrics.txt"),
        "groundtruth_metrics.txt"
    ]
    gt_data = {}
    for gt_path in gt_file_candidates:
        if os.path.exists(gt_path):
            try:
                with open(gt_path, 'r') as f:
                    for line in f:
                        if ":" in line:
                            parts = line.split(":", 1)
                            key = parts[0].strip()
                            try:
                                val = float(parts[1].strip())
                                if key == "CLIP Score": key = "CLIPScore"
                                if key not in ["Laplacian Variance", "SigLIP Score"]:
                                    gt_data[key] = val
                            except: pass
                if gt_data: break
            except: pass
    if gt_data: results["_GroundTruth_"] = gt_data

    if not results:
        print("No results found.")
        return

    df = pd.DataFrame.from_dict(results, orient='index')

    # Ordered based on User's request
    metrics_order = [
        "BLIPv2Score", "CLIPScore", "DINO v3 Score", "FGA_BLIP2Score",
        "FID Score", "HPSv2.1Score", "HPSv2Score", "ImageReward",
        "Inception Score", "MPS", "PickScore", "VQAScore"
    ]
    config_order = [
        "LLM_Model", "Retrieval_Method", "Retrieval_Datasets",
        "Image_Guidance", "Text_Guidance",
        "VAR_K", "DINO_L_Init", "DINO_L_Max",
        "TAC_Pass", "TAC_EarlyStop", "Resolution"
    ]

    # 1. Metric columns first
    final_cols = [m for m in metrics_order if m in df.columns]

    # 2. Config columns second
    for c in config_order:
        if c in df.columns: final_cols.append(c)

    # 3. Any unexpected leftovers
    remaining_cols = sorted([c for c in df.columns if c not in final_cols])
    final_cols.extend(remaining_cols)
    df = df[final_cols]
    df.sort_index(inplace=True)

    try:
        print("\n" + "="*120)
        print(df.to_markdown(floatfmt=".4f"))
        print("="*120)
    except ImportError:
        print("\n" + "="*120)
        print(df.to_string(float_format=lambda x: "{:.4f}".format(x) if x == x else "-"))
        print("="*120)

    csv_path = os.path.join(root_dir, "evalscope_results.csv")
    df.to_csv(csv_path)
    print(f"\n[INFO] Results exported to: {csv_path}")

    md_path = os.path.join(root_dir, "evalscope_results.md")
    try:
        with open(md_path, 'w') as f: f.write(df.to_markdown(floatfmt=".4f"))
        print(f"[INFO] Markdown exported to: {md_path}")
    except: pass

if __name__ == "__main__":
    main()
