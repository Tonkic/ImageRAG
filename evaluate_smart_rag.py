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
# --- 1. 参数定义 ---
# --------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluation: DINO v1/v2/v3 (All ViT-B) + KID")
parser.add_argument("--dataset_name", type=str, required=True, choices=['aircraft', 'cub', 'imagenet'])
parser.add_argument("--device_id", type=int, required=True)
parser.add_argument("--batch_size", type=int, default=32)

# --- DINOv3 路径配置 (已硬编码) ---
parser.add_argument("--dinov3_repo_path", type=str,
                    default="/home/tingyu/imageRAG/dinov3",
                    help="Path to the cloned dinov3 github repository")

parser.add_argument("--dinov3_weights_path", type=str,
                    default="/home/tingyu/imageRAG/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
                    help="Path to the local .pth weights file for DINOv3 ViT-B/16")

args = parser.parse_args()

# 设置显卡
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------------------------------
# --- 数据库配置 ---
# --------------------------------------------------
DATASET_CONFIGS = {
    "aircraft": {
        "classes_txt": "datasets/fgvc-aircraft-2013b/data/variants.txt",
        "real_images_list": "datasets/fgvc-aircraft-2013b/data/images_variant_test.txt",
        "real_images_root": "datasets/fgvc-aircraft-2013b/data/images",
        "results_dir": "results/Aircraft_SmartRAG"
    },
    "cub": {
        "classes_txt": "datasets/CUB_200_2011/classes.txt",
        "real_images_list": "datasets/CUB_200_2011/images.txt",
        "real_images_split_file": "datasets/CUB_200_2011/train_test_split.txt",
        "real_images_root": "datasets/CUB_200_2011/images",
        "results_dir": "results/CUB_SmartRAG"
    },
    "imagenet": {
        "classes_txt": "datasets/imagenet_classes.txt",
        "real_images_root": "datasets/ILSVRC2012_train",
        "results_dir": "results/ImageNet_SmartRAG"
    }
}

# --------------------------------------------------
# --- 2. 模型加载函数 (统一 ViT-B) ---
# --------------------------------------------------
def load_models(device):
    models = {}

    # --- 通用 DINO 预处理 (Resize 256 -> Crop 224) ---
    # 统一使用标准的 ImageNet 预处理，保证三个模型看到相同的输入
    models['dino_transform'] = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # ===========================
    # 1. DINO v3 (ViT-B/16) - Local
    # ===========================
    print(f"\n正在加载 DINO v3 (ViT-B/16 Local)...")
    print(f"  Repo: {args.dinov3_repo_path}")
    print(f"  Weights: {args.dinov3_weights_path}")

    sys.path.append(args.dinov3_repo_path)

    try:
        # 1. 先加载模型结构 (不传 weights 参数，防止报错或被忽略)
        models['dino_v3'] = torch.hub.load(
            repo_or_dir=args.dinov3_repo_path,
            model='dinov3_vitb16',
            source='local',
            pretrained=False
        )

        # 2. 手动加载权重文件
        checkpoint = torch.load(args.dinov3_weights_path, map_location='cpu')

        # DINOv3 的 checkpoint 通常包含 'model', 'teacher' 等 key
        # 我们需要提取真正的状态字典
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'teacher' in checkpoint:
            state_dict = checkpoint['teacher']
        elif 'student' in checkpoint:
            state_dict = checkpoint['student']
        else:
            state_dict = checkpoint # 假设直接是 state_dict

        # 3. 处理可能的 key 前缀不匹配 (例如 _orig_mod. 或 module.)
        # 创建一个新的 state_dict，去掉多余前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "").replace("_orig_mod.", "")
            new_state_dict[k] = v

        # 4. 加载权重 (strict=False 以容忍一些头部的缺失，但核心 backbone 必须匹配)
        msg = models['dino_v3'].load_state_dict(new_state_dict, strict=False)
        print(f"  -> DINO v3 权重加载结果: {msg}")

        models['dino_v3'] = models['dino_v3'].to(device).eval()
        print("  -> DINO v3 (Base) 加载并初始化成功!")

    except Exception as e:
        print(f"  -> DINO v3 加载失败: {e}")
        # 为了不中断脚本，如果加载失败，我们抛出异常或者退出
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ===========================
    # 2. DINO v2 (ViT-B/14) - Hub
    # ===========================
    # 注意：DINOv2 的 Base 模型标准是 vitb14 (patch size 14)
    print(f"正在加载 DINO v2 (dinov2_vitb14)...")
    models['dino_v2'] = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    models['dino_v2'] = models['dino_v2'].to(device).eval()

    # ===========================
    # 3. DINO v1 (ViT-B/16) - Hub
    # ===========================
    # 从 vits16 改为了 vitb16 以统一规模
    print(f"正在加载 DINO v1 (dino_vitb16)...")
    models['dino_v1'] = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    models['dino_v1'] = models['dino_v1'].to(device).eval()

    # ===========================
    # 4. OpenCLIP (ViT-B/32)
    # ===========================
    print(f"正在加载 OpenCLIP (ViT-B-32)...")
    models['clip_model'], _, models['clip_preprocess'] = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    models['clip_model'] = models['clip_model'].eval().to(device)
    models['clip_tokenizer'] = open_clip.get_tokenizer("ViT-B-32")

    # ===========================
    # 5. SigLIP
    # ===========================
    print(f"正在加载 SigLIP...")
    sig_name = 'hf-hub:timm/ViT-SO400M-14-SigLIP-384'
    models['siglip_model'], models['siglip_preprocess'] = create_model_from_pretrained(
        sig_name, device=device
    )
    models['siglip_model'] = models['siglip_model'].eval().to(device)
    models['siglip_tokenizer'] = get_tokenizer(sig_name)

    # ===========================
    # 6. KID
    # ===========================
    print(f"正在初始化 KID...")
    models['kid_v1'] = KernelInceptionDistance(subset_size=50, normalize=True).to(device)
    models['kid_final'] = KernelInceptionDistance(subset_size=50, normalize=True).to(device)
    models['kid_transform'] = T.Compose([T.Resize(299), T.CenterCrop(299), T.ToTensor()])

    return models

# --------------------------------------------------
# --- 3. 数据加载 ---
# --------------------------------------------------
def load_cub_data(config):
    print("Loading CUB data...")
    tasks = []
    class_map = {}
    try:
        with open(config['classes_txt']) as f:
            for l in f:
                parts = l.strip().split()
                class_map[int(parts[0])-1] = parts[1]
    except: return []

    img_paths = {}
    with open(config['real_images_list']) as f:
        for l in f:
            idx, p = l.strip().split()
            img_paths[int(idx)] = p

    img_labels = {}
    with open(os.path.join(os.path.dirname(config['classes_txt']), 'image_class_labels.txt')) as f:
        for l in f:
            idx, cls = l.strip().split()
            img_labels[int(idx)] = int(cls) - 1

    real_map = defaultdict(list)
    with open(config['real_images_split_file']) as f:
        for l in f:
            idx, is_train = l.strip().split()
            if is_train == '0': # Test set
                lbl = img_labels.get(int(idx))
                if lbl is not None:
                    full_p = os.path.join(config['real_images_root'], img_paths[int(idx)])
                    if os.path.exists(full_p): real_map[lbl].append(full_p)

    for lbl, name in class_map.items():
        simple = name.split('.')[-1].replace('_', ' ')
        tasks.append({
            "prompt": f"a photo of a {simple}",
            "safe_filename_prefix": f"{lbl:03d}_{name.split('.')[-1]}",
            "real_image_paths": real_map.get(lbl, [])
        })
    return tasks

def load_aircraft_data(config):
    print("Loading Aircraft data...")
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
                if os.path.exists(fp): real_map[class_map[p[1]]].append(fp)

    id_to_name = {v: k for k, v in class_map.items()}
    for i in range(len(class_map)):
        name = id_to_name[i]
        tasks.append({
            "prompt": f"a photo of a {name}",
            "safe_filename_prefix": f"{i:03d}_{name.replace(' ', '_').replace('/', '_')}",
            "real_image_paths": real_map[i]
        })
    return tasks

def load_imagenet_data(config):
    print("Loading ImageNet data...")
    tasks = []
    with open(config['classes_txt']) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            parts = line.split(':', 1)
            node = parts[0]
            name = parts[1].split(',')[0].strip()

            gt_dir = os.path.join(config['real_images_root'], node)
            paths = []
            if os.path.exists(gt_dir):
                paths = [os.path.join(gt_dir, x) for x in os.listdir(gt_dir) if x.endswith(('.jpg', '.jpeg', '.JPEG'))]

            tasks.append({
                "prompt": f"a photo of a {name}",
                "safe_filename_prefix": f"{i:03d}_{name.replace(' ', '_')}",
                "real_image_paths": paths
            })
    return tasks

def find_all_generated_versions(results_dir, prefix):
    versions = []
    regex_v = re.compile(rf"^{re.escape(prefix)}_(V(\d+)|V_FINAL_FIXED|V1).*\.(png|jpg|jpeg)$")
    if not os.path.isdir(results_dir): return []
    for f in os.listdir(results_dir):
        m = regex_v.match(f)
        if m:
            v_str = m.group(1)
            if v_str == "V1": num = 1
            elif v_str == "V_FINAL_FIXED": num = 999
            else: num = int(m.group(2))
            versions.append((num, os.path.join(results_dir, f)))
    versions.sort(key=lambda x: x[0])
    return versions

# --------------------------------------------------
# --- 4. 核心评估逻辑 ---
# --------------------------------------------------
def get_dino_features_batch(paths, model, transform, device, batch_size=32):
    """通用 Batch 特征提取"""
    feats = []
    # 最多取前 100 张真实图片以节省显存和时间
    eval_paths = paths[:100]

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

def evaluate_single_image(img_path, real_feats_map, txt_feats, models, device):
    scores = {}
    try:
        img = Image.open(img_path).convert("RGB")

        with torch.no_grad():
            dino_in = models['dino_transform'](img).unsqueeze(0).to(device)

            # 1. DINO v3 (ViT-B)
            v3_out = models['dino_v3'](dino_in)
            v3_out = F.normalize(v3_out, dim=-1)
            scores['dinov3_base'] = F.cosine_similarity(v3_out, real_feats_map['v3']).mean().item()

            # 2. DINO v2 (ViT-B)
            v2_out = models['dino_v2'](dino_in)
            v2_out = F.normalize(v2_out, dim=-1)
            scores['dinov2_base'] = F.cosine_similarity(v2_out, real_feats_map['v2']).mean().item()

            # 3. DINO v1 (ViT-B)
            v1_out = models['dino_v1'](dino_in)
            v1_out = F.normalize(v1_out, dim=-1)
            scores['dinov1_base'] = F.cosine_similarity(v1_out, real_feats_map['v1']).mean().item()

            # 4. CLIP & SigLIP
            c_in = models['clip_preprocess'](img).unsqueeze(0).to(device)
            c_feat = models['clip_model'].encode_image(c_in)
            c_feat /= c_feat.norm(dim=-1, keepdim=True)
            scores['clip'] = (c_feat @ txt_feats['clip'].T).item()

            s_in = models['siglip_preprocess'](img).unsqueeze(0).to(device)
            s_feat = models['siglip_model'].encode_image(s_in)
            s_feat = F.normalize(s_feat, dim=-1)
            scores['siglip'] = (s_feat @ txt_feats['siglip'].T).item()

    except Exception as e:
        print(f"Error eval {img_path}: {e}")
        return None
    return scores

# --------------------------------------------------
# --- 5. 主循环 ---
# --------------------------------------------------
def main():
    config = DATASET_CONFIGS[args.dataset_name]
    models = load_models(device)

    if args.dataset_name == 'cub': tasks = load_cub_data(config)
    elif args.dataset_name == 'aircraft': tasks = load_aircraft_data(config)
    elif args.dataset_name == 'imagenet': tasks = load_imagenet_data(config)
    else: tasks = []

    scores_base = []
    scores_final = []

    print(f"\n--- 开始评估 {len(tasks)} 个类别 ---")
    for task in tqdm(tasks):
        real_paths = task["real_image_paths"]
        if not real_paths: continue

        versions = find_all_generated_versions(config['results_dir'], task["safe_filename_prefix"])
        if not versions: continue

        # --- 1. 提取真实图像特征 (Batch) ---
        real_feats = {}
        real_feats['v3'] = get_dino_features_batch(real_paths, models['dino_v3'], models['dino_transform'], device)
        real_feats['v2'] = get_dino_features_batch(real_paths, models['dino_v2'], models['dino_transform'], device)
        real_feats['v1'] = get_dino_features_batch(real_paths, models['dino_v1'], models['dino_transform'], device)

        if any(v is None for v in real_feats.values()): continue

        # --- 2. 文本特征 ---
        txt_feats = {}
        with torch.no_grad():
            ctk = models['clip_tokenizer']([task["prompt"]]).to(device)
            txt_feats['clip'] = models['clip_model'].encode_text(ctk)
            txt_feats['clip'] /= txt_feats['clip'].norm(dim=-1, keepdim=True)

            stk = models['siglip_tokenizer']([task["prompt"]]).to(device)
            txt_feats['siglip'] = models['siglip_model'].encode_text(stk)
            txt_feats['siglip'] = F.normalize(txt_feats['siglip'], dim=-1)

        # --- 3. KID ---
        kp = random.sample(real_paths, min(len(real_paths), 50))
        kimgs = []
        for p in kp:
            try: kimgs.append(models['kid_transform'](Image.open(p).convert("RGB")))
            except: pass
        if kimgs:
            kt = torch.stack(kimgs).to(device)
            models['kid_v1'].update(kt, real=True)
            models['kid_final'].update(kt, real=True)

        # --- 4. 评估生成图 ---
        # Baseline
        res_base = evaluate_single_image(versions[0][1], real_feats, txt_feats, models, device)
        if res_base:
            scores_base.append(res_base)
            try: models['kid_v1'].update(models['kid_transform'](Image.open(versions[0][1]).convert("RGB")).unsqueeze(0).to(device), real=False)
            except: pass

        # Final
        res_final = evaluate_single_image(versions[-1][1], real_feats, txt_feats, models, device)
        if res_final:
            scores_final.append(res_final)
            try: models['kid_final'].update(models['kid_transform'](Image.open(versions[-1][1]).convert("RGB")).unsqueeze(0).to(device), real=False)
            except: pass

    # --- 报告 ---
    print("\nComputing KID...")
    try:
        kv1, _ = models['kid_v1'].compute()
        kfin, _ = models['kid_final'].compute()
    except:
        kv1, kfin = torch.tensor(0.0), torch.tensor(0.0)

    print("\n" + "="*70)
    print(f"  FINAL REPORT: {args.dataset_name} (All ViT-Base)")
    print("="*70)

    def show(name, lst, k):
        print(f"\n--- {name} ---")
        if not lst: return
        metrics = ['clip', 'siglip', 'dinov1_base', 'dinov2_base', 'dinov3_base']
        for m in metrics:
            print(f"{m.upper():<12}: {np.mean([x[m] for x in lst]):.4f}")
        print(f"{'KID':<12}: {k.item():.6f}")

    show("Baseline (V1)", scores_base, kv1)
    show("True Final", scores_final, kfin)
    print("="*70)

if __name__ == "__main__":
    main()