'''
智能 RAG 完整评估脚本
=====================================
为 smart_rag_dispatcher.py 评估所有结果。

功能:
1.  --dataset_name: 自动配置 'aircraft' 和 'cub' 的路径。
2.  报告 A (供参考): 评估所有中间迭代步骤 (V1, V2, V3...)。
3.  报告 B (公平比较): 评估 Baseline (V1) vs True Final (最终结果)。

用法:
python evaluate_smart_rag.py \
    --dataset_name aircraft \
    --device_id 0
'''

import os
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import open_clip
from open_clip import create_model_from_pretrained, get_tokenizer
import torchvision.transforms as T
import random
import numpy as np
import sys
import argparse
import re # <-- 用于智能排序

# --- 1. 配置 ---
# (我们将从 argparse 获取设备 ID)

# --------------------------------------------------
# --- 数据库配置中心 ---
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
    }
}

# --- 2. 模型加载函数 (不变) ---
def load_models(device):
    """加载所有评估所需的模型 (DINO, OpenCLIP, SigLIP)"""
    models = {}

    # a) 加载 DINO
    print(f"正在 {device} 上加载 DINO (vits16)...")
    models['dino_model'] = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    models['dino_model'] = models['dino_model'].eval().to(device)
    models['dino_transform'] = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # b) 加载 OpenCLIP (用于 CLIP Score)
    CLIP_MODEL = "ViT-B-32"
    CLIP_PRETRAINED = "laion2b_s34b_b79k"
    print(f"正在 {device} 上加载 OpenCLIP 模型 ({CLIP_MODEL})...")
    models['clip_model'], _, models['clip_preprocess'] = open_clip.create_model_and_transforms(
        CLIP_MODEL,
        pretrained=CLIP_PRETRAINED
    )
    models['clip_model'] = models['clip_model'].eval().to(device)
    models['clip_tokenizer'] = open_clip.get_tokenizer(CLIP_MODEL)

    # c) 加载 SigLIP
    SIGLIP_MODEL = 'hf-hub:timm/ViT-SO400M-14-SigLIP-384'
    print(f"正在 {device} 上加载 SigLIP 模型 ({SIGLIP_MODEL})...")
    models['siglip_model'], models['siglip_preprocess'] = create_model_from_pretrained(
        SIGLIP_MODEL,
        device=device
    )
    models['siglip_model'] = models['siglip_model'].eval().to(device)
    models['siglip_tokenizer'] = get_tokenizer(SIGLIP_MODEL)

    print("所有模型加载完毕。")
    return models

# --- 3. 数据集加载函数 (不变) ---

def load_cub_data(config):
    """加载 CUB 数据集的任务列表"""
    print(f"正在从 {config['classes_txt']} 加载 CUB 类别列表...")
    class_names_map = {} # {0: "001.Black_footed_Albatross", ...}

    try:
        with open(config['classes_txt']) as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 2: continue
                label_id = int(parts[0]) - 1 # 0-indexed
                class_name = parts[1]
                class_names_map[label_id] = class_name
    except FileNotFoundError:
        print(f"错误：找不到 {config['classes_txt']}。")
        sys.exit(1)

    print(f"正在从 {config['real_images_list']} 和 {config['real_images_split_file']} 映射 CUB 测试图像...")
    real_image_map = {label_id: [] for label_id in class_names_map}

    image_id_to_path = {}
    with open(config['real_images_list']) as f:
        for line in f.readlines():
            image_id, path = line.strip().split()
            image_id_to_path[int(image_id)] = path

    image_id_to_class = {}
    with open(os.path.join(os.path.dirname(config['classes_txt']), 'image_class_labels.txt')) as f:
        for line in f.readlines():
            image_id, class_id = line.strip().split()
            image_id_to_class[int(image_id)] = int(class_id) - 1 # 0-indexed

    with open(config['real_images_split_file']) as f:
        for line in f.readlines():
            image_id, is_train = line.strip().split()
            image_id = int(image_id)
            if is_train == '0': # '0' means test image
                label_id = image_id_to_class.get(image_id)
                if label_id is not None:
                    image_path = os.path.join(config['real_images_root'], image_id_to_path[image_id])
                    if os.path.exists(image_path):
                        real_image_map[label_id].append(image_path)

    print(f"找到了 {sum(len(v) for v in real_image_map.values())} 张真实测试图像。")

    # 构建任务列表
    tasks = []
    for label_id, full_class_name in class_names_map.items():
        simple_name = full_class_name.split('.')[-1].replace('_', ' ')
        tasks.append({
            "prompt": f"a photo of a {simple_name}",
            "safe_filename_prefix": f"{label_id:03d}_{full_class_name.split('.')[-1]}", # e.g., "000_Black_footed_Albatross"
            "real_image_paths": real_image_map.get(label_id, [])
        })
    return tasks

def load_aircraft_data(config):
    """加载 Aircraft 数据集的任务列表"""
    print(f"正在从 {config['classes_txt']} 加载 Aircraft 类别列表...")
    class_names_map = {} # {0: "Boeing 737-700", ...}
    try:
        with open(config['classes_txt']) as f:
            for i, line in enumerate(f.readlines()):
                full_class_name = line.strip()
                if full_class_name:
                    class_names_map[i] = full_class_name
    except FileNotFoundError:
        print(f"错误：找不到 {config['classes_txt']}。")
        sys.exit(1)

    print(f"正在从 {config['real_images_list']} 映射真实图像...")
    real_image_map = {label_id: [] for label_id in class_names_map}
    class_name_to_id = {name: id for id, name in class_names_map.items()}

    try:
        with open(config['real_images_list'], 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(' ', 1)
                if len(parts) != 2: continue
                image_id, class_name = parts[0], parts[1]

                if class_name in class_name_to_id:
                    label_id = class_name_to_id[class_name]
                    image_path = os.path.join(config['real_images_root'], f"{image_id}.jpg")
                    if not os.path.exists(image_path):
                        continue
                    real_image_map[label_id].append(image_path)
    except FileNotFoundError:
        print(f"错误：找不到 {config['real_images_list']}。")
        sys.exit(1)

    print(f"找到了 {sum(len(v) for v in real_image_map.values())} 张真实测试图像。")

    # 构建任务列表
    tasks = []
    for label_id, full_class_name in class_names_map.items():
        tasks.append({
            "prompt": f"a photo of a {full_class_name}",
            "safe_filename_prefix": f"{label_id:03d}_{full_class_name.replace(' ', '_').replace('/', '_')}", # e.g., "000_Boeing_737-700"
            "real_image_paths": real_image_map.get(label_id, [])
        })
    return tasks

# --- 4. 智能文件查找器 (不变) ---
def find_all_generated_versions(results_dir, prefix):
    """
    智能地查找所有迭代版本 (V1, V2, V3...) 并按顺序返回。
    返回一个 (version_num, path) 的元组列表。
    """
    versions = []
    # 匹配 V(数字) 或 V(FINAL_FIXED)
    regex_v = re.compile(rf"^{re.escape(prefix)}_(V(\d+)|V_FINAL_FIXED|V1_no_imageRAG).*\.(png|jpg|jpeg)$")

    if not os.path.isdir(results_dir):
        return []

    for f in os.listdir(results_dir):
        full_path = os.path.join(results_dir, f)

        match = regex_v.match(f)
        if match:
            version_str = match.group(1) # e.g., "V1", "V2", "V_FINAL_FIXED", "V1_no_imageRAG"

            if version_str == "V_FINAL_FIXED":
                version_num = 999 # 确保排在最后
            elif version_str == "V1_no_imageRAG":
                version_num = 1
            else:
                try:
                    version_num = int(match.group(2)) # e.g., "1", "2"
                except:
                    # 匹配 V1 (e.g. 000_707-320_V1.png)
                    if version_str == "V1":
                         version_num = 1
                    else:
                        continue # 匹配 V 但不是数字 (e.g., V_ICL_...)

            versions.append((version_num, full_path))

    # 去重并排序
    versions = list(set(versions)) # 处理 V1 和 V1_no_imageRAG 同时存在的情况
    versions.sort(key=lambda x: x[0]) # 按版本号 (1, 2, 3...) 升序排序
    return versions


# --- 5. 评估函数 (不变) ---
def evaluate_image(image_path, real_features, text_features, models, device):
    """对单个图像计算所有分数"""
    scores = {}
    try:
        gen_image_pil = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            # DINO (与真实图像比较)
            gen_dino_input = models['dino_transform'](gen_image_pil).unsqueeze(0).to(device)
            gen_features_dino = models['dino_model'](gen_dino_input)
            scores['dino'] = F.cosine_similarity(gen_features_dino, real_features['dino']).item()

            # CLIP (与文本比较)
            gen_clip_input = models['clip_preprocess'](gen_image_pil).unsqueeze(0).to(device)
            gen_features_clip = models['clip_model'].encode_image(gen_clip_input)
            gen_features_clip /= gen_features_clip.norm(dim=-1, keepdim=True)
            scores['clip'] = (gen_features_clip @ text_features['clip'].T).item()

            # SigLIP (与文本比较)
            gen_siglip_input = models['siglip_preprocess'](gen_image_pil).unsqueeze(0).to(device)
            gen_features_siglip = models['siglip_model'].encode_image(gen_siglip_input)
            gen_features_siglip = F.normalize(gen_features_siglip, dim=-1)
            scores['siglip'] = (gen_features_siglip @ text_features['siglip'].T).item()

    except Exception as e:
        print(f"评估 {image_path} 时出错: {e}")
        return None
    return scores

# --- 6. 主函数 (不变) ---
def main():
    parser = argparse.ArgumentParser(description="Unified Smart Evaluation Script for RAG (Full Report)")
    parser.add_argument("--dataset_name", type=str, required=True, choices=['cub', 'aircraft'], help="Dataset to evaluate.")
    parser.add_argument("--device_id", type=int, required=True, help="GPU device ID to use for evaluation.")

    args = parser.parse_args()

    # 获取配置
    if args.dataset_name not in DATASET_CONFIGS:
        print(f"错误: 数据集 '{args.dataset_name}' 未在 DATASET_CONFIGS 中配置。")
        sys.exit(1)
    config = DATASET_CONFIGS[args.dataset_name]
    results_dir = config['results_dir']

    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    device = "cuda"
    print(f"Setting CUDA_VISIBLE_DEVICES={args.device_id}")

    # 加载所有模型
    models = load_models(device)

    # --------------------------------------------------
    # --- ！！！分数存储 (报告A 和 报告B)！！！ ---
    # --------------------------------------------------
    # 1. 报告 A (供参考): 按版本分组
    scores_by_version = {}

    # 2. 报告 B (公平比较): 两个 100% 完整的列表
    scores_baseline_list = []
    scores_true_final_list = []
    # --------------------------------------------------

    # 根据选择的数据集加载任务
    if args.dataset_name == 'cub':
        tasks_to_evaluate = load_cub_data(config)
    elif args.dataset_name == 'aircraft':
        tasks_to_evaluate = load_aircraft_data(config)
    else:
        raise ValueError("Invalid dataset_name.")

    print(f"\n--- 开始评估 {args.dataset_name} | {len(tasks_to_evaluate)} 个类别 ---")
    print(f"结果目录: {results_dir}")

    # --- 评估循环 ---
    for task in tqdm(tasks_to_evaluate, desc=f"评估 {args.dataset_name} 图像"):
        try:
            prompt = task["prompt"]
            prefix = task["safe_filename_prefix"]

            if not task["real_image_paths"]:
                continue # 跳过没有真实图像的类别

            # 1. 准备真实图像和文本 (只需一次)
            real_image_path = random.choice(task["real_image_paths"])
            real_image_pil = Image.open(real_image_path).convert("RGB")

            real_features = {}
            text_features = {}
            with torch.no_grad():
                real_dino_input = models['dino_transform'](real_image_pil).unsqueeze(0).to(device)
                real_features['dino'] = models['dino_model'](real_dino_input)

                text_clip_input = models['clip_tokenizer']([prompt]).to(device)
                text_features['clip'] = models['clip_model'].encode_text(text_clip_input)
                text_features['clip'] /= text_features['clip'].norm(dim=-1, keepdim=True)

                text_features['clip'] = text_features['clip']

                text_siglip_input = models['siglip_tokenizer']([prompt]).to(device)
                text_features['siglip'] = models['siglip_model'].encode_text(text_siglip_input)
                text_features['siglip'] = F.normalize(text_features['siglip'], dim=-1)

            # --------------------------------------------------
            # --- ！！！公平评估逻辑！！！ ---
            # --------------------------------------------------

            # 2. 智能查找所有版本
            found_versions = find_all_generated_versions(results_dir, prefix)

            if not found_versions:
                continue

            # 3. 评估 V1 (Baseline)
            v1_path = found_versions[0][1] # V1 总是第一个
            v1_scores = evaluate_image(v1_path, real_features, text_features, models, device)

            if v1_scores:
                # 存入 Baseline (用于公平比较)
                scores_baseline_list.append(v1_scores)

            # 4. 评估 True Final (最终结果)
            final_path = found_versions[-1][1] # 取*最后一个*文件 (V_FINAL_FIXED 或 V4)
            final_scores = evaluate_image(final_path, real_features, text_features, models, device)

            if final_scores:
                # 存入 True Final (用于公平比较)
                scores_true_final_list.append(final_scores)

            # 5. 评估所有中间版本 (用于参考)
            cached_scores = {1: v1_scores, found_versions[-1][0]: final_scores}

            for version_num, image_path in found_versions:
                # 初始化该版本
                if version_num not in scores_by_version:
                    scores_by_version[version_num] = {'clip': [], 'siglip': [], 'dino': []}

                # 从缓存中获取分数 (避免重复计算 V1 和 VFinal)
                scores_to_add = cached_scores.get(version_num)

                if not scores_to_add: # 如果不是 V1 或 VFinal，则计算
                    scores_to_add = evaluate_image(image_path, real_features, text_features, models, device)

                # 存储分数
                if scores_to_add:
                    scores_by_version[version_num]['clip'].append(scores_to_add['clip'])
                    scores_by_version[version_num]['siglip'].append(scores_to_add['siglip'])
                    scores_by_version[version_num]['dino'].append(scores_to_add['dino'])
            # --------------------------------------------------

        except Exception as e:
            print(f"\n处理 {task['safe_filename_prefix']} 时出错: {e}")

    # --- 5. 显示最终结果 (!!! 关键修改 !!!) ---
    print(f"\n--- 评估完成 ({args.dataset_name}) ---")

    if not scores_by_version:
        print("未找到任何可评估的图像文件。请检查 'results_dir' 路径和文件名是否正确。")
        return

    # --- 报告 A: 迭代参考 (V1, V2...) ---
    print("\n--- 报告 A: 评估结果 (按迭代版本，供参考) ---")

    for version_num in sorted(scores_by_version.keys()):
        scores = scores_by_version[version_num]

        if version_num == 999:
            version_name = "V_FINAL_FIXED"
        else:
            version_name = f"V{version_num}"

        print(f"\n--- {version_name} (基于 {len(scores['clip'])} 张图像) ---")
        if len(scores['clip']) > 0:
            print(f"CLIP Score :   {np.mean(scores['clip']):.4f}")
            print(f"SigLIP Score : {np.mean(scores['siglip']):.4f}")
            print(f"DINO Score :   {np.mean(scores['dino']):.4f}")
        else:
            print("未找到该版本的分数。")

    # --- 报告 B: 公平比较 (V1 vs VFinal) ---
    print("\n\n--- 报告 B: 评估结果 (公平比较, 100% 任务) ---")

    print(f"\n--- Baseline (V1) (基于 {len(scores_baseline_list)} 张图像) ---")
    if scores_baseline_list:
        print(f"CLIP Score :   {np.mean([s['clip'] for s in scores_baseline_list]):.4f}")
        print(f"SigLIP Score : {np.mean([s['siglip'] for s in scores_baseline_list]):.4f}")
        print(f"DINO Score :   {np.mean([s['dino'] for s in scores_baseline_list]):.4f}")
    else:
        print("未找到 'V1' 图像。")

    print(f"\n--- True Final (RAG 修正后) (基于 {len(scores_true_final_list)} 张图像) ---")
    if scores_true_final_list:
        print(f"CLIP Score :   {np.mean([s['clip'] for s in scores_true_final_list]):.4f}")
        print(f"SigLIP Score : {np.mean([s['siglip'] for s in scores_true_final_list]):.4f}")
        print(f"DINO Score :   {np.mean([s['dino'] for s in scores_true_final_list]):.4f}")
    else:
        print("未找到 'True Final' 图像。")

if __name__ == "__main__":
    main()