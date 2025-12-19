'''
ZImage_FLUX_TAC_Aircraft.py
=============================
Configuration:
  - Stage 1: Input Interpreter (A_in)
  - Stage 2: Generation Engine (A_gen) -> Z-Image-Turbo
  - Stage 3: Quality Evaluator (A_eval) -> TAC (Qwen-VL)
  - Stage 4: Iterative Refinement -> FLUX.1-Fill-dev (Editing) or Z-Image (Regeneration)
  - Retrieval: Memory-Guided Retrieval (MGR)
  - Dataset: FGVC-Aircraft
'''

import argparse
import sys
import os
import time
import random
import traceback
import numpy as np
import torch
import openai
from PIL import Image
from tqdm import tqdm
import shutil
import base64
import json

# [IMPORTS]
from taxonomy_aware_critic import taxonomy_aware_diagnosis, input_interpreter, message_gpt
from memory_guided_retrieval import retrieve_img_per_caption, check_token_length
from global_memory import GlobalMemory
from diffusers import ZImagePipeline, FluxFillPipeline

# Proxy settings
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10000"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10000"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

parser = argparse.ArgumentParser(description="Z-Image + FLUX + TAC (Aircraft)")

parser.add_argument("--device_id", type=int, required=True)
parser.add_argument("--task_index", type=int, default=0)
parser.add_argument("--total_chunks", type=int, default=1)
parser.add_argument("--model_path", type=str, default="./Z-Image-Turbo")
parser.add_argument("--openai_api_key", type=str, required=True)
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/aircraft")
parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=["CLIP", "LongCLIP", "SigLIP", "ColPali", "Hybrid"], help="Retrieval Model")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
print(f"DEBUG: CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DATASET_CONFIG = {
    "classes_txt": "datasets/fgvc-aircraft-2013b/data/variants.txt",
    "train_list": "datasets/fgvc-aircraft-2013b/data/images_train.txt",
    "image_root": "datasets/fgvc-aircraft-2013b/data/images",
    "output_path": "results/ZImage_FLUX_TAC_MGR_Aircraft"
}

def setup_system():
    print("Loading Z-Image-Turbo...")
    # Load Z-Image (Generation Engine)
    # Assuming ZImagePipeline is available and works like Diffusers
    # Note: ZImagePipeline might need specific loading logic if it's custom
    # Here we assume it's a standard pipeline or wrapper
    try:
        pipe_gen = ZImagePipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, # [Fix] Enable for offloading
            local_files_only=True
        )
        # pipe_gen.to("cuda") # [Fix] Do not manually move to CUDA if using offload
        pipe_gen.enable_model_cpu_offload() # [Fix] Enable CPU offload for Z-Image too

        # Try enabling Flash Attention (as per Demo)
        try:
            pipe_gen.transformer.set_attention_backend("flash")
            print("✅ Flash Attention Enabled for Z-Image")
        except Exception:
            print("⚠️ Flash Attention Not Enabled for Z-Image")

    except Exception as e:
        print(f"Error loading Z-Image: {e}. Fallback to standard SDXL?")
        # Fallback logic if needed, or exit
        sys.exit(1)

    print("Loading FLUX.1-Fill-dev (Editing Engine)...")
    # Load FLUX (Editing Engine)
    # We load it to CPU first to save VRAM, move to CUDA only when needed
    pipe_edit = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
    pipe_edit.enable_model_cpu_offload()

    client = openai.OpenAI(
        api_key=args.openai_api_key,
        base_url="https://api.siliconflow.cn/v1/"
    )
    return pipe_gen, pipe_edit, client

def load_retrieval_db():
    print(f"Loading Aircraft DB...")
    paths = []
    with open(DATASET_CONFIG['train_list'], 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line: continue
            img_path = os.path.join(DATASET_CONFIG['image_root'], f"{line}.jpg")
            if os.path.exists(img_path):
                paths.append(img_path)
    print(f"Loaded {len(paths)} images.")
    return paths

def run_zimage(pipe, prompt, input_images, output_path, seed):
    torch.cuda.empty_cache() # [Fix] Clear VRAM before generation
    # Z-Image Generation Logic
    generator = torch.Generator(device="cuda").manual_seed(seed)
    # Z-Image Turbo settings: Low steps (e.g. 9), Guidance Scale 0.0
    image = pipe(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=generator
    ).images[0]
    image.save(output_path)

def run_flux_edit(pipe, prompt, image_path, mask_path, output_path, seed):
    torch.cuda.empty_cache() # [Fix] Clear VRAM before editing
    # FLUX Editing Logic
    generator = torch.Generator(device="cuda").manual_seed(seed)

    init_image = load_image(image_path)
    mask_image = load_image(mask_path)

    image = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        height=512,
        width=512,
        guidance_scale=30.0,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=generator
    ).images[0]
    image.save(output_path)

from diffusers.utils import load_image

def create_mask_from_bbox(image_path, bbox, output_mask_path):
    """
    bbox: [y_min, x_min, y_max, x_max] normalized 0-1000
    """
    img = Image.open(image_path)
    w, h = img.size

    mask = Image.new("L", (w, h), 0)

    if bbox:
        y_min, x_min, y_max, x_max = bbox
        # Normalize 1000 -> pixel coords
        x1 = int(x_min / 1000 * w)
        y1 = int(y_min / 1000 * h)
        x2 = int(x_max / 1000 * w)
        y2 = int(y_max / 1000 * h)

        # Draw white rectangle on black mask
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x1, y1, x2, y2], fill=255)

    mask.save(output_mask_path)
    return output_mask_path

if __name__ == "__main__":
    start_time = time.time()
    seed_everything(args.seed)

    # 1. Setup
    pipe_gen, pipe_edit, client = setup_system()
    retrieval_db = load_retrieval_db()
    os.makedirs(DATASET_CONFIG['output_path'], exist_ok=True)

    # Create logs directory
    logs_dir = os.path.join(DATASET_CONFIG['output_path'], "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Save Config
    config_path = os.path.join(logs_dir, "run_config.txt")
    with open(config_path, "w") as f:
        f.write("Run Configuration:\\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\\n")

    # 2. Load Tasks
    with open(DATASET_CONFIG['classes_txt'], 'r') as f:
        all_classes = [line.strip() for line in f.readlines() if line.strip()]
    my_classes = [c for i, c in enumerate(all_classes) if i % args.total_chunks == args.task_index]

    global_memory = GlobalMemory()

    for class_name in tqdm(my_classes):
        safe_name = class_name.replace(" ", "_").replace("/", "-")
        raw_prompt = f"a photo of a {class_name}"

        log_file = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}.log")
        f_log = open(log_file, "w")
        f_log.write(f"Raw Prompt: {raw_prompt}\\n")

        # --- Stage 1: Input Interpreter (A_in) ---
        f_log.write("--- Stage 1: Input Interpreter ---\\n")
        analysis_report = input_interpreter(raw_prompt, client, args.llm_model)
        detailed_prompt = analysis_report.get("detailed_prompt", raw_prompt)
        f_log.write(f"Detailed Prompt: {detailed_prompt}\\n")
        f_log.write(f"Report: {json.dumps(analysis_report, indent=2)}\\n")

        # --- Stage 2: Generation Engine (A_gen) ---
        f_log.write("--- Stage 2: Generation Engine (Z-Image) ---\\n")
        v1_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V1.png")

        if not os.path.exists(v1_path):
            run_zimage(pipe_gen, detailed_prompt, [], v1_path, args.seed)

        current_image = v1_path
        current_prompt = detailed_prompt
        retry_cnt = 0
        best_score = -1
        best_image_path = current_image

        final_success_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_FINAL.png")

        while retry_cnt < args.max_retries:
            f_log.write(f"\\n--- Iteration {retry_cnt+1} ---\\n")

            # --- Stage 3: Quality Evaluator (A_eval) ---
            # [Knowledge Retrieval] - Sanity Check (Lazy Load if not present)
            if 'reference_specs' not in locals():
                from taxonomy_aware_critic import generate_knowledge_specs
                try:
                    reference_specs = generate_knowledge_specs(class_name, client, args.llm_model)
                    f_log.write(f"Reference Specs: {reference_specs}\\n")
                except Exception as e:
                    f_log.write(f"Reference Specs Retrieval Failed: {e}\\n")
                    reference_specs = None

            diagnosis = taxonomy_aware_diagnosis(current_prompt, [current_image], client, args.llm_model, reference_specs=reference_specs)

            score = diagnosis.get('final_score', 0)
            taxonomy_status = diagnosis.get('taxonomy_check', 'unknown')
            critique = diagnosis.get('critique', '')
            refined_prompt = diagnosis.get('refined_prompt', current_prompt)
            error_analysis = diagnosis.get('error_analysis', {})
            error_type = error_analysis.get('type', 'Global')
            bbox = error_analysis.get('bbox', None)

            f_log.write(f"Score: {score} | Taxonomy: {taxonomy_status} | Error Type: {error_type}\nCritique: {critique}\n")

            if score > best_score:
                best_score = score
                best_image_path = current_image

            if score >= 8.0 or (score >= 6.0 and taxonomy_status == 'correct'):
                f_log.write(f">> Success! (Score: {score}, Taxonomy: {taxonomy_status})\n")
                shutil.copy(current_image, final_success_path)
                break

            # --- Stage 4: Iterative Refinement ---
            next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")

            if error_type == "Local" and bbox:
                # Use FLUX for Editing (Inpainting)
                f_log.write(f">> Strategy: Local Editing with FLUX (BBox: {bbox})\\n")
                mask_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+1}_mask.png")
                create_mask_from_bbox(current_image, bbox, mask_path)

                # Use refined prompt for editing
                run_flux_edit(pipe_edit, refined_prompt, current_image, mask_path, next_path, args.seed + retry_cnt)

            else:
                # Use Z-Image for Regeneration (Global/Structural fix) with Semantic RAG
                f_log.write(">> Strategy: Global Regeneration with Z-Image (Semantic RAG)\\n")

                # 1. 检索参考图 (Retrieval)
                # Force Class Name injection
                query_text = f"{class_name} {class_name}. {refined_prompt}"

                # 检索最相关的 1-3 张图
                retrieved_lists, _ = retrieve_img_per_caption(
                    [query_text],
                    retrieval_db,
                    embeddings_path=args.embeddings_path,
                    k=2,
                    device="cuda",
                    method=args.retrieval_method
                )
                retrieved_paths = retrieved_lists[0] if retrieved_lists else []

                visual_cues = ""
                if retrieved_paths:
                    f_log.write(f"   [RAG] Retrieved references: {len(retrieved_paths)}\\n")

                    # 2. 视觉转译 (Visual Translation via VLM)
                    # 构造一个让 VLM "看图说话" 的 Prompt
                    extraction_msg = (
                        "You are an aircraft taxonomy expert. "
                        "Identify the specific visual distinguishing features of the aircraft in these reference images. "
                        "Focus ONLY on: Fuselage length (short/long), Engine shape and position, Wing placement, and Tail details. "
                        "Do not describe the background. "
                        "Summarize these features into a single concise text description string."
                    )

                    # 调用现有的 message_gpt 函数 (它支持 image_paths 参数)
                    # 注意：args.llm_model 必须是支持视觉的模型 (如 Qwen-VL)
                    try:
                        visual_cues = message_gpt(
                            extraction_msg,
                            client,
                            image_paths=retrieved_paths,
                            model=args.llm_model,
                            max_retries=2
                        )
                        f_log.write(f"   [RAG] Extracted Cues: {visual_cues}\\n")
                    except Exception as e:
                        print(f"VLM Extraction failed: {e}")

                # 3. 提示词增强 (Prompt Injection)
                # 将提取到的视觉特征拼接到 refined_prompt 后面
                if visual_cues:
                    # 构造新的 Prompt，增加权重强调
                    rag_prompt = f"{refined_prompt}. Visual characteristics: {visual_cues}"
                else:
                    rag_prompt = refined_prompt

                f_log.write(f"   [Gen] RAG Prompt: {rag_prompt}\\n")

                # 4. 执行生成 (Standard Text-to-Image)
                # 此时使用的是包含丰富细节的纯文本 Prompt
                run_zimage(
                    pipe_gen,
                    rag_prompt, # 使用增强后的 Prompt
                    [],         # input_images 为空，因为 ZImage 不支持
                    next_path,
                    args.seed + retry_cnt + 1
                )

            current_image = next_path
            current_prompt = refined_prompt
            retry_cnt += 1

        # Final Save
        if not os.path.exists(final_success_path) and best_image_path:
             shutil.copy(best_image_path, final_success_path)

        f_log.close()
