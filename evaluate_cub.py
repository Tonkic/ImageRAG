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

# --- 1. 配置 ---
DEVICE = "cuda:2"
CLASSES_TXT = "CUB_200_2011/classes.txt"
REAL_IMAGE_BASE_DIR = "datasets/CUB_test"
GENERATED_IMAGE_DIR = "results/CUB"
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
SIGLIP_MODEL = 'hf-hub:timm/ViT-SO400M-14-SigLIP-384'

# RAG 增强后图像的文件名参数 (来自 imageRAG_OmniGen_3gpu.py)
GUIDANCE_SCALE = 2.5
IMAGE_GUIDANCE_SCALE = 1.6

# --- 2. 加载评估模型 ---

# a) 加载 DINO
print(f"正在 {DEVICE} 上加载 DINO (vits16)...")
dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
dino_model = dino_model.eval().to(DEVICE)
dino_transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# b) 加载 OpenCLIP (用于 CLIP Score)
print(f"正在 {DEVICE} 上加载 OpenCLIP 模型 ({CLIP_MODEL})...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL,
    pretrained=CLIP_PRETRAINED
)
clip_model = clip_model.eval().to(DEVICE)
clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

# c) 加载 SigLIP
print(f"正在 {DEVICE} 上加载 SigLIP 模型 ({SIGLIP_MODEL})...")
siglip_model, siglip_preprocess = create_model_from_pretrained(
    SIGLIP_MODEL,
    device=DEVICE
)
siglip_model = siglip_model.eval().to(DEVICE)
siglip_tokenizer = get_tokenizer(SIGLIP_MODEL)
print("所有模型加载完毕。")

# --- 3. 加载类别列表 ---
print(f"正在从 {CLASSES_TXT} 加载类别列表...")
class_names_map = {} # {0: "001.Black...", 1: "002.Laysan..."}
with open(CLASSES_TXT) as f:
    for line in f.readlines():
        parts = line.strip().split()
        label_id_zero_based = int(parts[0]) - 1 # 0-索引
        full_class_name = parts[1]
        class_names_map[label_id_zero_based] = full_class_name
print(f"找到了 {len(class_names_map)} 个类别。")

# --- 4. 运行评估循环 ---

# 为 "no RAG" 和 "RAG" 分别创建分数列表
scores_no_rag = {'clip': [], 'siglip': [], 'dino': []}
scores_rag = {'clip': [], 'siglip': [], 'dino': []}

# 映射真实图像（不变）
real_image_map = {}
print(f"正在从 {REAL_IMAGE_BASE_DIR} 映射真实图像...")
for class_name in os.listdir(REAL_IMAGE_BASE_DIR):
    class_dir = os.path.join(REAL_IMAGE_BASE_DIR, class_name)
    if os.path.isdir(class_dir):
        try:
            label_id = int(class_name.split('.')[0]) - 1
            real_image_map[label_id] = [
                os.path.join(class_dir, f) for f in os.listdir(class_dir)
            ]
        except ValueError:
            pass
print(f"找到了 {len(real_image_map)} 个类别的真实图像。")

unique_labels_to_evaluate = list(class_names_map.keys())

for label_id in tqdm(unique_labels_to_evaluate, desc="评估 CUB 图像"):
    try:
        full_class_name = class_names_map[label_id]

        # --- 1. 准备 Prompt 和 真实图像 ---
        simple_name = full_class_name.split('.')[-1].replace('_', ' ')
        prompt = f"a photo of a {simple_name}"

        # 获取真实图像 (用于 DINO)
        if label_id not in real_image_map or not real_image_map[label_id]:
            continue # 如果测试集中没有该类别的图像，则跳过

        real_image_path = random.choice(real_image_map[label_id])
        real_image_pil = Image.open(real_image_path).convert("RGB")

        # --- 2. 预处理 真实图像 和 文本 (只需一次) ---
        with torch.no_grad():
            real_dino_input = dino_transform(real_image_pil).unsqueeze(0).to(DEVICE)
            real_features_dino = dino_model(real_dino_input)

            text_clip_input = clip_tokenizer([prompt]).to(DEVICE)
            text_features_clip = clip_model.encode_text(text_clip_input)
            text_features_clip /= text_features_clip.norm(dim=-1, keepdim=True)

            text_siglip_input = siglip_tokenizer([prompt]).to(DEVICE)
            text_features_siglip = siglip_model.encode_text(text_siglip_input)
            text_features_siglip = F.normalize(text_features_siglip, dim=-1)

        # --- 3. 查找并评估 "no RAG" 图像 ---
        no_rag_filename = f"{label_id:03d}_{full_class_name}_no_imageRAG.png"
        no_rag_image_path = os.path.join(GENERATED_IMAGE_DIR, no_rag_filename)

        if os.path.exists(no_rag_image_path):
            gen_image_pil = Image.open(no_rag_image_path).convert("RGB")
            with torch.no_grad():
                # DINO
                gen_dino_input = dino_transform(gen_image_pil).unsqueeze(0).to(DEVICE)
                gen_features_dino = dino_model(gen_dino_input)
                scores_no_rag['dino'].append(F.cosine_similarity(gen_features_dino, real_features_dino).item())

                # CLIP
                gen_clip_input = clip_preprocess(gen_image_pil).unsqueeze(0).to(DEVICE)
                gen_features_clip = clip_model.encode_image(gen_clip_input)
                gen_features_clip /= gen_features_clip.norm(dim=-1, keepdim=True)
                scores_no_rag['clip'].append((gen_features_clip @ text_features_clip.T).item())

                # SigLIP
                gen_siglip_input = siglip_preprocess(gen_image_pil).unsqueeze(0).to(DEVICE)
                gen_features_siglip = siglip_model.encode_image(gen_siglip_input)
                gen_features_siglip = F.normalize(gen_features_siglip, dim=-1)
                scores_no_rag['siglip'].append((gen_features_siglip @ text_features_siglip.T).item())

        # --- 4. 查找并评估 "RAG" 图像 ---
        rag_filename = f"{label_id:03d}_{full_class_name}_gs_{GUIDANCE_SCALE}_im_gs_{IMAGE_GUIDANCE_SCALE}.png"
        rag_image_path = os.path.join(GENERATED_IMAGE_DIR, rag_filename)

        if os.path.exists(rag_image_path):
            gen_image_pil = Image.open(rag_image_path).convert("RGB")
            with torch.no_grad():
                # DINO
                gen_dino_input = dino_transform(gen_image_pil).unsqueeze(0).to(DEVICE)
                gen_features_dino = dino_model(gen_dino_input)
                scores_rag['dino'].append(F.cosine_similarity(gen_features_dino, real_features_dino).item())

                # CLIP
                gen_clip_input = clip_preprocess(gen_image_pil).unsqueeze(0).to(DEVICE)
                gen_features_clip = clip_model.encode_image(gen_clip_input)
                gen_features_clip /= gen_features_clip.norm(dim=-1, keepdim=True)
                scores_rag['clip'].append((gen_features_clip @ text_features_clip.T).item())

                # SigLIP
                gen_siglip_input = siglip_preprocess(gen_image_pil).unsqueeze(0).to(DEVICE)
                gen_features_siglip = siglip_model.encode_image(gen_siglip_input)
                gen_features_siglip = F.normalize(gen_features_siglip, dim=-1)
                scores_rag['siglip'].append((gen_features_siglip @ text_features_siglip.T).item())

    except Exception as e:
        print(f"\n处理 {full_class_name} 时出错: {e}")

# --- 5. 显示最终结果 ---
print("\n--- 评估完成 ---")

print("\n--- 评估结果 (no RAG) ---")
if len(scores_no_rag['clip']) > 0:
    print(f"CLIP Score :   {np.mean(scores_no_rag['clip']):.4f}")
    print(f"SigLIP Score : {np.mean(scores_no_rag['siglip']):.4f}")
    print(f"DINO Score :   {np.mean(scores_no_rag['dino']):.4f}")
    print(f"\n(基于 {len(scores_no_rag['clip'])} / {len(unique_labels_to_evaluate)} 个已找到的 'no RAG' 图像)")
else:
    print("未找到 'no RAG' 图像 (例如: *_no_imageRAG.png)")

print("\n--- 评估结果 (RAG后) ---")
if len(scores_rag['clip']) > 0:
    print(f"CLIP Score :   {np.mean(scores_rag['clip']):.4f}")
    print(f"SigLIP Score : {np.mean(scores_rag['siglip']):.4f}")
    print(f"DINO Score :   {np.mean(scores_rag['dino']):.4f}")
    print(f"\n(基于 {len(scores_rag['clip'])} / {len(unique_labels_to_evaluate)} 个已找到的 'RAG' 图像)")
else:
    print("未找到 'RAG' 图像 (例如: *_gs_2.5_im_gs_1.6.png)")