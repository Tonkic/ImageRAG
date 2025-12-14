'''
FLUX_TAC_MGR_Aircraft.py
=============================
Configuration:
  - Generator: FLUX.1-dev
  - Critic: Taxonomy-Aware Critic (TAC) -> Fine-grained diagnosis
  - Retrieval: Memory-Guided Retrieval (MGR) -> Dynamic RAG with Exclusion List
  - Dataset: FGVC-Aircraft

Usage:
  python FLUX_TAC_MGR_Aircraft.py \
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

# [IMPORTS] Custom Modules
from taxonomy_aware_critic import taxonomy_aware_diagnosis # TAC Logic
from memory_guided_retrieval import retrieve_img_per_caption
from global_memory import GlobalMemory # MGR Logic
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from PIL import Image, ImageDraw

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="FLUX + TAC + MGR (Aircraft)")

# Core Config
parser.add_argument("--device_id", type=int, required=True, help="GPU Device ID")
parser.add_argument("--task_index", type=int, default=0)
parser.add_argument("--total_chunks", type=int, default=1)

# Paths
parser.add_argument("--openai_api_key", type=str, required=True)
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")

# Generation Params
parser.add_argument("--seed", type=int, default=0, help="Global Random Seed")
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--text_guidance_scale", type=float, default=30.0) # FLUX Fill default
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/aircraft")

args = parser.parse_args()

# Environment
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print(f"DEBUG: CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"DEBUG: Torch sees {torch.cuda.device_count()} devices. Current device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})")

# --- 2. Reproducibility (Seed Fix) ---
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[System] Global seed set to: {seed}")

# --- 3. Config ---
DATASET_CONFIG = {
    "classes_txt": "datasets/fgvc-aircraft-2013b/data/variants.txt",
    "train_list": "datasets/fgvc-aircraft-2013b/data/images_train.txt",
    "image_root": "datasets/fgvc-aircraft-2013b/data/images",
    "output_path": "results/FLUX_TAC_MGR_Aircraft"
}

# --- 4. Setup System ---
def setup_system():
    print("Loading FLUX.1-Fill-dev...")
    pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload() # Save VRAM

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
    # FLUX Fill requires image + mask.
    # If input_images (RAG) is provided, we use it as the base image.
    # We create a mask that covers the center (e.g. 80%) to allow the model to regenerate the object
    # while keeping the context/style from the border.

    generator = torch.Generator("cpu").manual_seed(seed)
    clean_prompt = prompt.replace("<|image_1|>", "").strip()

    width, height = 512, 512

    if input_images and len(input_images) > 0:
        # Use RAG image
        rag_img_path = input_images[0]
        if isinstance(rag_img_path, str):
            base_image = load_image(rag_img_path)
        else:
            base_image = rag_img_path

        base_image = base_image.convert("RGB").resize((width, height))

        # Create Mask: Mask the center 80%
        mask = Image.new("L", (width, height), 0) # 0 = Keep
        draw = ImageDraw.Draw(mask)
        margin = int(width * 0.1) # 10% margin
        draw.rectangle([margin, margin, width - margin, height - margin], fill=255) # 255 = Mask/Regenerate
    else:
        # No RAG image (Fallback): Create black image + Full Mask
        # This effectively acts as T2I
        base_image = Image.new("RGB", (width, height), (0, 0, 0))
        mask = Image.new("L", (width, height), 255) # Mask everything

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
    image.save(output_path)# --- 5. Main Loop ---
if __name__ == "__main__":
    # 1. Set Seed
    seed_everything(args.seed)

    # 2. Load DB & Pre-calculate Embeddings (BEFORE loading FLUX)
    retrieval_db = load_retrieval_db()

    print("Pre-calculating/Loading retrieval embeddings on GPU...")
    try:
        # Run a dummy retrieval on the FULL database to force caching of all images
        retrieve_img_per_caption(
            ["warmup_query"],
            retrieval_db,
            embeddings_path=args.embeddings_path,
            k=1,
            device="cuda",
            method='Hybrid'
        )
        torch.cuda.empty_cache()
        print("Retrieval embeddings cached successfully.")
    except Exception as e:
        print(f"Warning during embedding caching: {e}")

    # 3. Init FLUX
    pipe, client = setup_system()
    os.makedirs(DATASET_CONFIG['output_path'], exist_ok=True)

    # 4. Load Tasks
    with open(DATASET_CONFIG['classes_txt'], 'r') as f:
        all_classes = [line.strip() for line in f.readlines() if line.strip()]

    my_classes = [c for i, c in enumerate(all_classes) if i % args.total_chunks == args.task_index]
    print(f"Processing {len(my_classes)} classes.")

    for class_name in tqdm(my_classes):
        safe_name = class_name.replace(" ", "_").replace("/", "-")
        prompt = f"a photo of a {class_name}"

        log_file = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}.log")
        final_success_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_FINAL.png")
        f_log = open(log_file, "w")
        f_log.write(f"Prompt: {prompt}\n")

        # Phase 1: Initial
        v1_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V1.png")

        # [Optimization] Shared Baseline Logic
        baseline_dir = "results/FLUX_Baseline_Aircraft"
        baseline_v1_path = os.path.join(baseline_dir, f"{safe_name}_V1.png")

        if not os.path.exists(v1_path):
            if os.path.exists(baseline_v1_path):
                shutil.copy(baseline_v1_path, v1_path)
            else:
                run_flux(pipe, prompt, [], v1_path, args.seed)
                # Try to populate baseline
                try:
                    os.makedirs(baseline_dir, exist_ok=True)
                    shutil.copy(v1_path, baseline_v1_path)
                except: pass

        current_image = v1_path
        current_prompt = prompt
        retry_cnt = 0

        # [MGR Core]: Global Memory for Re-ranking
        global_memory = GlobalMemory()
        last_used_ref = None

        # [Score Tracking]
        best_score = -1
        best_image_path = None

        while retry_cnt < args.max_retries:
            f_log.write(f"\n--- Retry {retry_cnt+1} ---\n")

            # A. TAC Diagnosis
            diagnosis = taxonomy_aware_diagnosis(current_prompt, [current_image], client, args.llm_model)

            score = diagnosis.get('final_score', 0)
            taxonomy_status = diagnosis.get('taxonomy_check', 'unknown')
            critique = diagnosis.get('critique', '')
            refined_prompt = diagnosis.get('refined_prompt', current_prompt)
            mgr_queries = diagnosis.get('retrieval_queries', [class_name])

            f_log.write(f"Decision: Score {score} | Taxonomy: {taxonomy_status}\nCritique: {critique}\n")

            # Update Best
            if score > best_score:
                best_score = score
                best_image_path = current_image

            if score >= 8.0:
                f_log.write(f">> Success! (Score: {score})\n")
                shutil.copy(current_image, final_success_path)
                break

            # B. Memory-Guided Retrieval
            # Use the specific queries from TAC
            query_text = " ".join(mgr_queries)
            if len(query_text) > 300: query_text = query_text[:300]

            try:
                retrieved_lists, retrieved_scores = retrieve_img_per_caption(
                    [query_text], retrieval_db,
                    embeddings_path=args.embeddings_path,
                    k=50, device="cpu", method='Hybrid',
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
            f_log.write(f">> Ref: {best_ref} (Score: {best_ref_score:.4f})\n")

            # C. Generation Strategy
            next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")

            # FLUX doesn't use image guidance scale in this T2I mode, but we log strategy
            if score < 6.0:
                f_log.write(">> Strategy: Fix Taxonomy (Prompt Refinement)\n")
            else:
                f_log.write(">> Strategy: Optimize Details (Prompt Refinement)\n")

            # Use refined prompt. FLUX ignores reference image.
            gen_prompt = refined_prompt
            run_flux(pipe, gen_prompt, [best_ref], next_path, args.seed + retry_cnt + 1)

            current_image = next_path
            current_prompt = refined_prompt
            retry_cnt += 1

        # Final Check
        if not os.path.exists(final_success_path):
            f_log.write(f"\n--- Final Check (Last Generated) ---\n")
            # Evaluate the last image generated (current_image)
            if os.path.exists(current_image):
                diagnosis = taxonomy_aware_diagnosis(current_prompt, [current_image], client, args.llm_model)
                score = diagnosis.get('score', 0)
                f_log.write(f"Final Image Score: {score}\n")
                if score > best_score:
                    best_score = score
                    best_image_path = current_image

            f_log.write(f">> Loop finished. Best Score: {best_score}. Saving best image to FINAL.\n")
            if best_image_path and os.path.exists(best_image_path):
                shutil.copy(best_image_path, final_success_path)

        f_log.close()

    # --- End of Class Loop ---
    print("\n============================================")
    print("All classes processed.")
