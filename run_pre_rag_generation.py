'''
python evaluate_pre_rag.py \
--dataset_name aircraft \
--device_id 0 \
--results_dir results/PreRAG
'''

import argparse
import sys
import os

# --- 关键：在顶部设置环境变量 ---
parser = argparse.ArgumentParser(description="OmniGen2 Pre-RAG Pipeline (Scheme 2)")
# ... (所有参数) ...
parser.add_argument("--omnigen2_path", type=str, default="./OmniGen2", help="Path to the OmniGen2 checkout/folder")
parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2", help="Hugging Face ID")
parser.add_argument("--transformer_lora_path", type=str, default=None, help="Path to the LoRA weights (e.g., ./OmniGen2-EditScore7B)")
parser.add_argument("--openai_api_key", type=str, required=True, help="SiliconFlow API Key")
parser.add_argument("--llm_model", type=str, default="Pro/Qwen/Qwen2.5-VL-7B-Instruct", help="The name of the VLM model")
parser.add_argument("--cpu_offload_mode", type=str, default="none", choices=['none', 'model', 'sequential'],
                    help="CPU Offload mode for OmniGen2. 'none' for direct GPU.")

# --- 关键：新参数，用于选择数据集 ---
parser.add_argument("--dataset_name", type=str, required=True, choices=['cub', 'aircraft'], help="Dataset to process (cub or aircraft)")
# -----------------------------------

parser.add_argument("--out_path", type=str, default="results/PreRAG", help="Directory to save generated images and logs")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--text_guidance_scale", type=float, default=7.5, help="CFG for text")
parser.add_argument("--image_guidance_scale", type=float, default=3.0, help="Guidance for reference image (higher for consistency)")
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--data_lim", type=int, default=-1)
parser.add_argument("--embeddings_path", type=str, default="")
parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=['CLIP', 'SigLIP', 'MoE', 'gpt-rerank'])
parser.add_argument("--device_id", type=int, required=True, help="Actual GPU device ID to use (e.g., 0, 1, or 3)")
parser.add_argument("--task_index", type=int, required=True, help="The index of the task chunk (e.g., 0, 1, or 2)")
parser.add_argument("--total_chunks", type=int, default=1, help="Total number of chunks to split the job into (e.g., 3)")

args = parser.parse_args()

# 立即设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
print(f"Setting CUDA_VISIBLE_DEVICES={args.device_id}")
# ------------------------------

# 现在安全导入
import openai
import numpy as np
from tqdm import tqdm
from retrieval import *
from utils import * # <-- 确保 message_gpt 在 utils.py 中
from PIL import Image
import torch

# --- 辅助函数：运行 OmniGen2 (不变) ---
def run_omnigen2(prompt, images_list, out_path, args, pipe, device):
    print(f"running OmniGen2 inference... (Prompt: {prompt[:50]}...)")
    pil_images = [Image.open(p) for p in images_list] if images_list else []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    images = pipe(
        prompt=prompt,
        input_images=pil_images,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        num_inference_steps=50,
        height=args.height,
        width=args.width,
        negative_prompt="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        num_images_per_prompt=1,
        generator=generator,
        output_type="pil"
    ).images
    images[0].save(out_path)
    print(f"  [Success] 已保存图像: {out_path}")
# ------------------------------

# 主程序入口
if __name__ == "__main__":

    # (Args 已经被解析过了)

    # --- 1. 脚本启动时的一次性设置 ---

    script_dir = os.path.dirname(os.path.abspath(__file__))
    omnigen2_abs_path = os.path.abspath(os.path.join(script_dir, args.omnigen2_path))

    print(f"正在将 OmniGen2 仓库路径 {omnigen2_abs_path} 添加到 sys.path")
    sys.path.append(omnigen2_abs_path)

    try:
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
    except ImportError as e:
        print(f"错误：无法从 '{args.omnigen2_path}' 导入 OmniGen2 核心组件。")
        print(f"详细错误: {e}")
        sys.exit(1)

    client = openai.OpenAI(
        api_key=args.openai_api_key,
        base_url="https://api.siliconflow.cn/v1/"
    )
    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    device = "cuda"
    os.makedirs(args.out_path, exist_ok=True)

    # --- 2. 加载模型（只加载 OmniGen2） ---

    print(f"[Device {args.device_id}] Loading OmniGen2 model (ONCE) from {args.omnigen2_model_path}...")
    if args.transformer_lora_path:
        print(f"[Device {args.device_id}] Applying LoRA weights from: {args.transformer_lora_path}")

    pipe = OmniGen2Pipeline.from_pretrained(
        args.omnigen2_model_path,
        torch_dtype=torch.bfloat16,
        transformer_lora_path=args.transformer_lora_path,
        trust_remote_code=True
    )

    print(f"[Device {args.device_id}] Loading Transformer component separately...")
    transformer = OmniGen2Transformer2DModel.from_pretrained(
        args.omnigen2_model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    pipe.transformer = transformer

    if args.cpu_offload_mode == 'model':
        pipe.enable_model_cpu_offload()
        print(f"[Device {args.device_id}] OmniGen2 model loaded with 'model' CPU Offload enabled.")
    elif args.cpu_offload_mode == 'sequential':
        pipe.enable_sequential_cpu_offload()
        print(f"[Device {args.device_id}] OmniGen2 model loaded with 'sequential' CPU Offload (VRAM < 3GB) enabled.")
    else:
        pipe.to(device)
        print(f"[Device {args.device_id}] OmniGen2 model loaded directly onto device: {device}.")

    # --- 3. 加载 RAG 数据库路径 (动态) ---
    retrieval_image_paths = []

    if args.dataset_name == 'cub':
        data_path = "datasets/CUB_train"
        print(f"[Device {args.device_id}] 正在递归扫描 {data_path} 中的所有图像...")
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    retrieval_image_paths.append(os.path.join(root, file))

    elif args.dataset_name == 'aircraft':
        data_path = "datasets/fgvc-aircraft-2013b/data"
        train_list_file = os.path.join(data_path, "images_train.txt")
        print(f"[Device {args.device_id}] 正在从 {train_list_file} 加载 RAG 数据库...")
        with open(train_list_file, 'r') as f:
            for image_id in f.readlines():
                image_id = image_id.strip()
                if image_id:
                    image_path = os.path.join(data_path, "images", f"{image_id}.jpg")
                    if os.path.exists(image_path):
                        retrieval_image_paths.append(image_path)

    print(f"[Device {args.device_id}] 已找到 {len(retrieval_image_paths)} 张图像用于检索。")
    if not retrieval_image_paths:
        print(f"错误：在 {data_path} 中未找到图像。请检查路径。")
        sys.exit(1)

    if args.data_lim != -1:
        retrieval_image_paths = retrieval_image_paths[:args.data_lim]
    embeddings_path = args.embeddings_path or f"datasets/embeddings/{args.dataset_name}_train"

    # --- 4. 加载并分配任务列表 (动态) ---
    all_items_to_generate = []

    if args.dataset_name == 'cub':
        classes_txt = "datasets/CUB_200_2011/classes.txt"
        print(f"[Device {args.device_id}] 正在从 {classes_txt} 加载 CUB 类别列表...")
        with open(classes_txt) as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 2: continue
                label_id = int(parts[0]) - 1
                class_name = parts[1]
                prompt = f"a photo of a {class_name.split('.')[-1].replace('_', ' ')}"
                out_name = f"{label_id:03d}_{class_name}"
                all_items_to_generate.append((prompt, out_name))

    elif args.dataset_name == 'aircraft':
        classes_txt = "datasets/fgvc-aircraft-2013b/data/variants.txt"
        print(f"[Device {args.device_id}] 正在从 {classes_txt} 加载 Aircraft 类别列表...")
        with open(classes_txt) as f:
            for i, line in enumerate(f.readlines()):
                class_name = line.strip()
                if class_name:
                    label_id = i
                    prompt = f"a photo of a {class_name}"
                    safe_name = class_name.replace(' ', '_').replace('/', '_')
                    out_name = f"{label_id:03d}_{safe_name}"
                    all_items_to_generate.append((prompt, out_name))

    items_for_this_gpu = []
    for i, item in enumerate(all_items_to_generate):
        if i % args.total_chunks == args.task_index:
            items_for_this_gpu.append(item)
    print(f"[Device {args.device_id}] 总共 {len(all_items_to_generate)} 个类别。此设备 (Task {args.task_index}) 将处理 {len(items_for_this_gpu)} 个。")

    # --- 5. 主循环 (Pre-RAG 逻辑) ---
    for current_prompt, current_out_name in tqdm(items_for_this_gpu, desc=f"在 Device {args.device_id} (Task {args.task_index}) 上生成图像"):

        print(f"\n--- [Device {args.device_id}] 正在运行: {current_prompt} ---")

        out_txt_file = os.path.join(args.out_path, current_out_name + ".txt")
        f = open(out_txt_file, "w")
        f.write(f"prompt: {current_prompt}\n")

        # --- 步骤 1: VLM Pre-Check (无图像) ---
        # (我们使用一个简化的 utils.py 函数，只调用 VLM)
        vlm_prompt = f"Is the concept in the prompt '{current_prompt}' a common, simple item (like 'a dog' or 'a tree') OR a rare, specific, fine-grained concept (like 'a Boeing 737-700' or 'a Black-footed Albatross')? Answer only 'Simple' or 'Rare'."

        try:
            vlm_decision = message_gpt(vlm_prompt, client, model=args.llm_model)
            print(f"  [VLM Pre-Check] VLM 判定: {vlm_decision}")
        except Exception as e:
            print(f"  [VLM Error] VLM 预检查失败: {e}. 默认为 'Rare'.")
            vlm_decision = "Rare"

        # --- 步骤 2: RAG 检索 (如果需要) ---
        images_list = []
        final_prompt = current_prompt

        if 'Simple' in vlm_decision:
            f.write("VLM deemed prompt simple. Skipping RAG.\n")

        else: # (VLM 判定为 "Rare" 或 "Rare concept")
            f.write(f"VLM deemed prompt rare ({vlm_decision}). Running RAG.\n")

            # 步骤 2a: VLM 生成检索标题
            # (我们只使用原始提示作为检索标题，这更简单且通常有效)
            captions = [current_prompt]
            f.write(f"captions: {captions}\n")

            # 步骤 2b: RAG 检索
            paths = retrieve_img_per_caption(captions, retrieval_image_paths, embeddings_path=embeddings_path,
                                             k=1, device=device, method=args.retrieval_method)

            if not paths or not paths[0]:
                f.write("RAG retrieval failed to find images. Skipping context.")
            else:
                reference_image_path = paths[0][0]
                images_list = [reference_image_path] # <-- 这是我们的 RAG 上下文

                # 步骤 2c: 创建上下文提示
                final_prompt = f"Using the provided reference image as visual context, generate: {current_prompt}"
                f.write(f"final retrieved path (as reference): {reference_image_path}\n")
                f.write(f"final_prompt: {final_prompt}\n")

        # --- 步骤 3: 最终生成 (只调用一次) ---

        out_name_final = f"{current_out_name}_PreRAG.png"
        out_path_final = os.path.join(args.out_path, out_name_final)
        f.write(f"running OmniGen2 (Final Generation), will save result to: {out_path_final}\n")

        if os.path.exists(out_path_final):
             f.write("Output file already exists. Skipping generation.\n")
             f.close()
             continue

        run_omnigen2(
            prompt=final_prompt,
            images_list=images_list, # 要么为空 (T2I), 要么包含参考图像 (上下文)
            out_path=out_path_final,
            args=args,
            pipe=pipe,
            device=device
        )
        f.close()

    print(f"--- [Device {args.device_id}] 已完成所有 {len(items_for_this_gpu)} 个任务 ---")