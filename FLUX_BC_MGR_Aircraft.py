'''
FLUX_BC_MGR_Aircraft.py
=============================
Configuration:
  - Generator: FLUX.1-dev
  - Critic: Binary Critic (BC)
  - Retrieval: Memory-Guided Retrieval (MGR)
  - Dataset: FGVC-Aircraft

Usage:
  python FLUX_BC_MGR_Aircraft.py \
      --device_id 0 \
      --task_index 0 \
      --total_chunks 1 \
      --openai_api_key "sk-..." \
      --seed 42
'''

import argparse
import sys
import os
import json
import shutil
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import random
import openai
import clip

# [IMPORTS]
from binary_critic import retrieval_caption_generation  # Critic
from memory_guided_retrieval import retrieve_img_per_caption
from global_memory import GlobalMemory # MGR Logic
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from PIL import Image, ImageDraw

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="FLUX + BC + MGR (Aircraft)")

# Core Config
parser.add_argument("--device_id", type=int, required=True)
parser.add_argument("--task_index", type=int, default=0)
parser.add_argument("--total_chunks", type=int, default=1)

# Paths
parser.add_argument("--openai_api_key", type=str, required=True)
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")

# Params
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--text_guidance_scale", type=float, default=30.0)
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/aircraft")
parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=["CLIP", "LongCLIP", "SigLIP", "ColPali", "Hybrid"], help="Retrieval Model")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
print(f"DEBUG: CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"DEBUG: Torch sees {torch.cuda.device_count()} devices. Current device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 2. Config ---
DATASET_CONFIG = {
    "classes_txt": "datasets/fgvc-aircraft-2013b/data/variants.txt",
    "train_list": "datasets/fgvc-aircraft-2013b/data/images_train.txt",
    "image_root": "datasets/fgvc-aircraft-2013b/data/images",
    "output_path": "results/FLUX_BC_MGR_Aircraft"
}

# --- 3. Setup ---
def setup_system():
    print("Loading FLUX.1-Fill-dev...")
    pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    client = openai.OpenAI(
        api_key=args.openai_api_key,
        base_url="https://api.siliconflow.cn/v1/"
    )
    return pipe, client

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

def run_flux(pipe, prompt, input_images, output_path, seed):
    generator = torch.Generator("cpu").manual_seed(seed)
    clean_prompt = prompt.replace("<|image_1|>", "").strip()

    width, height = 512, 512

    if input_images and len(input_images) > 0:
        rag_img_path = input_images[0]
        if isinstance(rag_img_path, str):
            base_image = load_image(rag_img_path)
        else:
            base_image = rag_img_path
        base_image = base_image.convert("RGB").resize((width, height))

        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        margin = int(width * 0.1)
        draw.rectangle([margin, margin, width - margin, height - margin], fill=255)
    else:
        base_image = Image.new("RGB", (width, height), (0, 0, 0))
        mask = Image.new("L", (width, height), 255)

    image = pipe(
        prompt=clean_prompt,
        image=base_image,
        mask_image=mask,
        height=height,
        width=width,
        guidance_scale=args.text_guidance_scale,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=generator
    ).images[0]
    image.save(output_path)

if __name__ == "__main__":
    seed_everything(args.seed)

    retrieval_db = load_retrieval_db()

    # Pre-calc embeddings
    print("Pre-calculating/Loading retrieval embeddings on GPU...")
    try:
        retrieve_img_per_caption(
            ["warmup_query"],
            retrieval_db,
            embeddings_path=args.embeddings_path,
            k=1,
            device="cuda",
            method=args.retrieval_method
        )
        torch.cuda.empty_cache()
        print("Retrieval embeddings cached successfully.")
    except Exception as e:
        print(f"Warning during embedding caching: {e}")

    pipe, client = setup_system()
    os.makedirs(DATASET_CONFIG['output_path'], exist_ok=True)

    with open(DATASET_CONFIG['classes_txt'], 'r') as f:
        all_classes = [line.strip() for line in f.readlines() if line.strip()]

    my_tasks = [c for i, c in enumerate(all_classes) if i % args.total_chunks == args.task_index]
    print(f"Processing {len(my_tasks)} classes.")

    for class_name in tqdm(my_tasks):
        safe_name = class_name.replace(" ", "_").replace("/", "-")
        prompt = f"a photo of a {class_name}"

        log_file = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}.log")
        final_success_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_FINAL.png")
        f_log = open(log_file, "w")
        f_log.write(f"Prompt: {prompt}\n")

        v1_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V1.png")
        baseline_dir = "results/FLUX_Baseline_Aircraft"
        baseline_v1_path = os.path.join(baseline_dir, f"{safe_name}_V1.png")

        if not os.path.exists(v1_path):
            if os.path.exists(baseline_v1_path):
                shutil.copy(baseline_v1_path, v1_path)
            else:
                run_flux(pipe, prompt, [], v1_path, args.seed)
                try:
                    os.makedirs(baseline_dir, exist_ok=True)
                    shutil.copy(v1_path, baseline_v1_path)
                except: pass

        current_image = v1_path
        retry_cnt = 0
        global_memory = GlobalMemory()
        last_used_ref = None

        while retry_cnt < args.max_retries:
            f_log.write(f"\n--- Retry {retry_cnt+1} ---\n")

            # 1. Critic (Binary)
            diagnosis = retrieval_caption_generation(
                prompt, [current_image],
                gpt_client=client, model=args.llm_model
            )
            status = diagnosis.get('status')
            f_log.write(f"Decision: {status}\n")

            # [MGR Feedback Loop]
            if last_used_ref is not None:
                is_match = (status == 'success')
                global_memory.add_feedback(last_used_ref, prompt, is_match=is_match)
                f_log.write(f"  [MGR] Feedback recorded for {os.path.basename(last_used_ref)}: {'Match' if is_match else 'Mismatch'}\n")

            if status == 'success':
                f_log.write(">> Success!\n")
                shutil.copy(current_image, final_success_path)
                break

            # 2. Retrieval (MGR) - Log only for FLUX
            try:
                retrieved_lists, retrieved_scores = retrieve_img_per_caption(
                    [prompt], retrieval_db,
                    embeddings_path=args.embeddings_path,
                    k=50, device="cuda:0", method=args.retrieval_method,
                    global_memory=global_memory
                )
                candidates = retrieved_lists[0]
                candidate_scores = retrieved_scores[0]
            except RuntimeError as e:
                f_log.write(f">> Retrieval Error: {e}\n")
                candidates = []

            if not candidates:
                f_log.write(">> No references found.\n")
                break

            best_ref = candidates[0]
            best_ref_score = candidate_scores[0]
            global_memory.add(best_ref)
            last_used_ref = best_ref
            f_log.write(f">> Ref (Unused): {best_ref} (Score: {best_ref_score:.4f})\n")

            # 3. Generation (New Seed)
            next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")
            f_log.write(">> Strategy: Retry with new seed\n")

            run_flux(pipe, prompt, [last_used_ref] if last_used_ref else [], next_path, args.seed + retry_cnt + 1)

            current_image = next_path
            retry_cnt += 1

        # Final Check
        if not os.path.exists(final_success_path):
            f_log.write(f"\n--- Final Check (Last Generated) ---\n")
            if os.path.exists(current_image):
                diagnosis = retrieval_caption_generation(
                    prompt, [current_image],
                    gpt_client=client, model=args.llm_model
                )
                status = diagnosis.get('status')
                f_log.write(f"Final Status: {status}\n")
                if status == 'success':
                    shutil.copy(current_image, final_success_path)

        f_log.close()

    print("\n============================================")
    print("All classes processed.")
