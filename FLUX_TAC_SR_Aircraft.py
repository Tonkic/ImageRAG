'''
FLUX_TAC_SR_Aircraft.py
=============================
Configuration:
  - Generator: FLUX.1-dev
  - Critic: Taxonomy-Aware Critic (TAC) -> Fine-grained diagnosis
  - Retrieval: Static Retrieval (SR) -> No memory/exclusion list
  - Dataset: FGVC-Aircraft

Usage:
  python FLUX_TAC_SR_Aircraft.py \
      --device_id 0 \
      --task_index 0 \
      --total_chunks 1 \
      --openai_api_key "sk-..."
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
from taxonomy_aware_critic import taxonomy_aware_diagnosis # The new Critic
from static_retrieval import retrieve_img_per_caption      # The Static Retrieval logic
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from PIL import Image, ImageDraw

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="FLUX + TAC + SR (Aircraft)")

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
parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=["CLIP", "LongCLIP", "SigLIP", "SigLIP2", "ColPali", "Qwen2.5-VL", "Qwen3-VL"], help="Retrieval Model")

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
    "output_path": "results/FLUX_TAC_SR_Aircraft"
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
    image.save(output_path)if __name__ == "__main__":
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
            device="cuda"
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
        current_prompt = prompt
        retry_cnt = 0
        best_score = -1
        best_image_path = None

        while retry_cnt < args.max_retries:
            f_log.write(f"\n--- Retry {retry_cnt+1} ---\n")

            diagnosis = taxonomy_aware_diagnosis(current_prompt, [current_image], client, args.llm_model)

            score = diagnosis.get('final_score', 0)
            taxonomy_status = diagnosis.get('taxonomy_check', 'unknown')
            critique = diagnosis.get('critique', '')
            refined_prompt = diagnosis.get('refined_prompt', current_prompt)

            f_log.write(f"Decision: Score {score} | Taxonomy: {taxonomy_status}\nCritique: {critique}\n")

            if score > best_score:
                best_score = score
                best_image_path = current_image

            if score >= 8.0:
                f_log.write(f">> Success! (Score: {score})\n")
                shutil.copy(current_image, final_success_path)
                break

            # B. Static Retrieval (Simplified for SR)
            query = refined_prompt
            if len(query) > 300: query = query[:300]

            try:
                retrieved_lists, retrieved_scores = retrieve_img_per_caption(
                    [query], retrieval_db,
                    embeddings_path=args.embeddings_path,
                    k=1, device="cuda", method=args.retrieval_method
                )
                if retrieved_lists and retrieved_lists[0]:
                    best_ref = retrieved_lists[0][0]
                    best_ref_score = retrieved_scores[0][0]
                else:
                    raise ValueError("Empty retrieval result")
            except Exception as e:
                f_log.write(f">> Retrieval Error: {e}\n")
                # Fallback
                try:
                    retrieved_lists, retrieved_scores = retrieve_img_per_caption(
                        [prompt], retrieval_db,
                        embeddings_path=args.embeddings_path,
                        k=1, device="cuda", method=args.retrieval_method
                    )
                    if retrieved_lists and retrieved_lists[0]:
                        best_ref = retrieved_lists[0][0]
                        best_ref_score = retrieved_scores[0][0]
                    else:
                        raise ValueError("Empty retrieval result in fallback")
                except Exception as e2:
                    f_log.write(f">> Retrieval Error (Fallback): {e2}\n")
                    # Final Fallback: Random
                    import random
                    if retrieval_db:
                        best_ref = random.choice(retrieval_db)
                        best_ref_score = 0.0
                        f_log.write(f">> Retrieval Failed completely. Using Random image: {best_ref}\n")
                    else:
                        f_log.write(f">> Retrieval Failed and DB empty. Skipping generation.\n")
                        continue

            f_log.write(f">> Ref: {best_ref} (Score: {best_ref_score:.4f})\n")

            next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")

            if score < 6.0:
                f_log.write(">> Strategy: Fix Taxonomy (Prompt Refinement)\n")
            else:
                f_log.write(">> Strategy: Optimize Details (Prompt Refinement)\n")

            gen_prompt = refined_prompt
            run_flux(pipe, gen_prompt, [best_ref], next_path, args.seed + retry_cnt + 1)

            current_image = next_path
            current_prompt = refined_prompt
            retry_cnt += 1

        if not os.path.exists(final_success_path):
            f_log.write(f"\n--- Final Check (Last Generated) ---\n")
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

    print("\n============================================")
    print("All classes processed.")
