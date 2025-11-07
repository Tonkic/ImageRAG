'''
python evaluate_all.py \
--dataset_name cub \
--device_id 0 \
--results_dir results/CUB_OmniGen2_LoRA \
--cub_classes_txt CUB_200_2011/classes.txt

python evaluate_all.py \
--dataset_name aircraft \
--device_id 0 \
--results_dir results/Aircraft_OmniGen2_LoRA
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

# --- 1. 配置 ---
# (我们将从 argparse 获取设备 ID)

# --- 2. 模型加载函数 ---
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

# --- 3. 数据集加载函数 ---

def load_cub_data(args):
    """加载 CUB 数据集的任务列表"""
    print(f"正在从 {args.cub_classes_txt} 加载 CUB 类别列表...")
    class_names_map = {} # {0: "001.Black...", 1: "002.Laysan..."}
    try:
        with open(args.cub_classes_txt) as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 2: continue
                label_id = int(parts[0]) - 1
                class_name = parts[1]
                class_names_map[label_id] = class_name
    except FileNotFoundError:
        print(f"错误：找不到 {args.cub_classes_txt}。")
        print(f"       检查的路径: {os.path.abspath(args.cub_classes_txt)}")
        sys.exit(1)

    print(f"正在从 {args.cub_real_images_dir} 映射真实图像...")
    real_image_map = {}
    for class_name in os.listdir(args.cub_real_images_dir):
        class_dir = os.path.join(args.cub_real_images_dir, class_name)
        if os.path.isdir(class_dir):
            try:
                label_id = int(class_name.split('.')[0]) - 1
                real_image_map[label_id] = [
                    os.path.join(class_dir, f) for f in os.listdir(class_dir)
                ]
            except ValueError:
                pass
    print(f"找到了 {len(real_image_map)} 个类别的真实图像。")

    # 构建任务列表
    tasks = []
    for label_id, full_class_name in class_names_map.items():
        simple_name = full_class_name.split('.')[-1].replace('_', ' ')
        tasks.append({
            "prompt": f"a photo of a {simple_name}",
            "safe_filename": f"{label_id:03d}_{full_class_name}", # e.g., "000_001.Black_footed_Albatross"
            "real_image_paths": real_image_map.get(label_id, [])
        })
    return tasks

def load_aircraft_data(args):
    """加载 Aircraft 数据集的任务列表"""
    print(f"正在从 {args.aircraft_classes_txt} 加载 Aircraft 类别列表...")
    class_names_map = {} # {0: "Boeing 737-700", ...}
    try:
        with open(args.aircraft_classes_txt) as f:
            for i, line in enumerate(f.readlines()):
                full_class_name = line.strip()
                if full_class_name:
                    class_names_map[i] = full_class_name
    except FileNotFoundError:
        print(f"错误：找不到 {args.aircraft_classes_txt}。")
        sys.exit(1)

    print(f"正在从 {args.aircraft_data_dir} 映射真实图像...")
    real_image_map = {}
    class_name_to_id = {name: id for id, name in class_names_map.items()}
    test_labels_file = os.path.join(args.aircraft_data_dir, "images_variant_test.txt")

    try:
        with open(test_labels_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(' ', 1)
                if len(parts) != 2: continue
                image_id, class_name = parts[0], parts[1]

                if class_name in class_name_to_id:
                    label_id = class_name_to_id[class_name]
                    image_path = os.path.join(args.aircraft_data_dir, "images", f"{image_id}.jpg")
                    if not os.path.exists(image_path):
                        continue
                    if label_id not in real_image_map:
                        real_image_map[label_id] = []
                    real_image_map[label_id].append(image_path)
    except FileNotFoundError:
        print(f"错误：找不到 {test_labels_file}。")
        sys.exit(1)

    print(f"找到了 {len(real_image_map)} 个类别的真实图像。")
    # --- 真实图像映射结束 ---

    # 构建任务列表
    tasks = []
    for label_id, full_class_name in class_names_map.items():
        tasks.append({
            "prompt": f"a photo of a {full_class_name}",
            "safe_filename": f"{label_id:03d}_{full_class_name.replace(' ', '_').replace('/', '_')}", # e.g., "000_Boeing_737-700"
            "real_image_paths": real_image_map.get(label_id, [])
        })
    return tasks

# --- 4. 主函数 ---
def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation Script for RAG vs No-RAG")
    parser.add_argument("--dataset_name", type=str, required=True, choices=['cub', 'aircraft'], help="Dataset to evaluate.")
    parser.add_argument("--device_id", type=int, required=True, help="GPU device ID to use for evaluation.")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the results folder (e.g., results/CUB_OmniGen2_LoRA)")

    # --- 关键修复：CUB 路径现在指向 'datasets/' 内部 ---
    parser.add_argument("--cub_classes_txt", type=str, default="datasets/CUB_200_2011/classes.txt", help="Path for CUB classes.txt")
    parser.add_argument("--cub_real_images_dir", type=str, default="datasets/CUB_test", help="Path for CUB_test real images")
    # ----------------------------------------------------------------

    parser.add_argument("--aircraft_data_dir", type=str, default="datasets/fgvc-aircraft-2013b/data", help="Path for aircraft 'data' folder")
    parser.add_argument("--aircraft_classes_txt", type=str, default="datasets/fgvc-aircraft-2013b/data/variants.txt", help="Path for aircraft 'variants.txt'")

    args = parser.parse_args()

    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    device = "cuda"
    print(f"Setting CUDA_VISIBLE_DEVICES={args.device_id}")

    # 加载所有模型
    models = load_models(device)

    # 为 "no RAG" 和 "RAG" 分别创建分数列表
    scores_no_rag = {'clip': [], 'siglip': [], 'dino': []}
    scores_rag = {'clip': [], 'siglip': [], 'dino': []}

    # 根据选择的数据集加载任务
    if args.dataset_name == 'cub':
        tasks_to_evaluate = load_cub_data(args)
    elif args.dataset_name == 'aircraft':
        tasks_to_evaluate = load_aircraft_data(args)
    else:
        raise ValueError("Invalid dataset_name. Choose 'cub' or 'aircraft'.")

    print(f"\n--- 开始评估 {args.dataset_name} | {len(tasks_to_evaluate)} 个类别 ---")

    # --- 评估循环 ---
    for task in tqdm(tasks_to_evaluate, desc=f"评估 {args.dataset_name} 图像"):
        try:
            prompt = task["prompt"]
            safe_filename = task["safe_filename"]

            if not task["real_image_paths"]:
                continue # 跳过没有真实图像的类别

            # 1. 准备真实图像和文本 (只需一次)
            real_image_path = random.choice(task["real_image_paths"])
            real_image_pil = Image.open(real_image_path).convert("RGB")

            with torch.no_grad():
                real_dino_input = models['dino_transform'](real_image_pil).unsqueeze(0).to(device)
                real_features_dino = models['dino_model'](real_dino_input)

                text_clip_input = models['clip_tokenizer']([prompt]).to(device)
                text_features_clip = models['clip_model'].encode_text(text_clip_input)
                text_features_clip /= text_features_clip.norm(dim=-1, keepdim=True)

                text_siglip_input = models['siglip_tokenizer']([prompt]).to(device)
                text_features_siglip = models['siglip_model'].encode_text(text_siglip_input)
                text_features_siglip = F.normalize(text_features_siglip, dim=-1)

            # 2. 查找并评估 "no RAG" 图像 (e.g., ..._no_imageRAG.png)
            no_rag_filename = f"{safe_filename}_no_imageRAG.png"
            no_rag_image_path = os.path.join(args.results_dir, no_rag_filename)

            if os.path.exists(no_rag_image_path):
                gen_image_pil = Image.open(no_rag_image_path).convert("RGB")
                with torch.no_grad():
                    # DINO
                    gen_dino_input = models['dino_transform'](gen_image_pil).unsqueeze(0).to(device)
                    gen_features_dino = models['dino_model'](gen_dino_input)
                    scores_no_rag['dino'].append(F.cosine_similarity(gen_features_dino, real_features_dino).item())
                    # CLIP
                    gen_clip_input = models['clip_preprocess'](gen_image_pil).unsqueeze(0).to(device)
                    gen_features_clip = models['clip_model'].encode_image(gen_clip_input)
                    gen_features_clip /= gen_features_clip.norm(dim=-1, keepdim=True)
                    scores_no_rag['clip'].append((gen_features_clip @ text_features_clip.T).item())
                    # SigLIP
                    gen_siglip_input = models['siglip_preprocess'](gen_image_pil).unsqueeze(0).to(device)
                    gen_features_siglip = models['siglip_model'].encode_image(gen_siglip_input)
                    gen_features_siglip = F.normalize(gen_features_siglip, dim=-1)
                    scores_no_rag['siglip'].append((gen_features_siglip @ text_features_siglip.T).item())

            # 3. 查找并评估 "RAG" 图像 (e.g., ..._EDITED.png)
            rag_filename = f"{safe_filename}_EDITED.png"
            rag_image_path = os.path.join(args.results_dir, rag_filename)

            if os.path.exists(rag_image_path):
                gen_image_pil = Image.open(rag_image_path).convert("RGB")
                with torch.no_grad():
                    # DINO
                    gen_dino_input = models['dino_transform'](gen_image_pil).unsqueeze(0).to(device)
                    gen_features_dino = models['dino_model'](gen_dino_input)
                    scores_rag['dino'].append(F.cosine_similarity(gen_features_dino, real_features_dino).item())
                    # CLIP
                    gen_clip_input = models['clip_preprocess'](gen_image_pil).unsqueeze(0).to(device)
                    gen_features_clip = models['clip_model'].encode_image(gen_clip_input)
                    gen_features_clip /= gen_features_clip.norm(dim=-1, keepdim=True)
                    scores_rag['clip'].append((gen_features_clip @ text_features_clip.T).item())
                    # SigLIP
                    gen_siglip_input = models['siglip_preprocess'](gen_image_pil).unsqueeze(0).to(device)
                    gen_features_siglip = models['siglip_model'].encode_image(gen_siglip_input)
                    gen_features_siglip = F.normalize(gen_features_siglip, dim=-1)
                    scores_rag['siglip'].append((gen_features_siglip @ text_features_siglip.T).item())

        except Exception as e:
            print(f"\n处理 {task['safe_filename']} 时出错: {e}")

    # --- 5. 显示最终结果 ---
    print(f"\n--- 评估完成 ({args.dataset_name}) ---")

    print(f"\n--- 评估结果 (初始生成 / no RAG) ---")
    if len(scores_no_rag['clip']) > 0:
        print(f"CLIP Score :   {np.mean(scores_no_rag['clip']):.4f}")
        print(f"SigLIP Score : {np.mean(scores_no_rag['siglip']):.4f}")
        print(f"DINO Score :   {np.mean(scores_no_rag['dino']):.4f}")
        print(f"\n(基于 {len(scores_no_rag['clip'])} / {len(tasks_to_evaluate)} 个已找到的 'no RAG' 图像)")
    else:
        print("未找到 'no RAG' 图像 (例如: *_no_imageRAG.png)")

    print(f"\n--- 评估结果 (RAG后 / EDITED) ---")
    if len(scores_rag['clip']) > 0:
        print(f"CLIP Score :   {np.mean(scores_rag['clip']):.4f}")
        print(f"SigLIP Score : {np.mean(scores_rag['siglip']):.4f}")
        print(f"DINO Score :   {np.mean(scores_rag['dino']):.4f}")
        print(f"\n(基于 {len(scores_rag['clip'])} / {len(tasks_to_evaluate)} 个已找到的 'RAG' 图像)")
    else:
        print("未找到 'RAG' 图像 (例如: *_EDITED.png)")

if __name__ == "__main__":
    main()