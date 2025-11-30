'''
OmniGenV2_TAC_MGR_CUB.py
=============================
Configuration:
  - Generator: OmniGen V2
  - Critic: Taxonomy-Aware Critic (TAC) -> Fine-grained diagnosis
  - Retrieval: Memory-Guided Retrieval (MGR) -> Dynamic RAG with Exclusion List
  - Dataset: CUB-200-2011

Usage:
  python OmniGenV2_TAC_MGR_CUB.py \
      --device_id 0 \
      --task_index 0 \
      --total_chunks 1 \
      --omnigen2_path ./OmniGen2 \
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

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="OmniGenV2 + TAC + MGR (CUB)")

# Core Config
parser.add_argument("--device_id", type=int, required=True, help="GPU Device ID")
parser.add_argument("--task_index", type=int, default=0)
parser.add_argument("--total_chunks", type=int, default=1)

# Paths
parser.add_argument("--omnigen2_path", type=str, default="./OmniGen2")
parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2")
parser.add_argument("--transformer_lora_path", type=str, default="OmniGen2-EditScore7B" if os.path.exists("OmniGen2-EditScore7B") else None)
parser.add_argument("--openai_api_key", type=str, required=True)
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")

# Generation Params
parser.add_argument("--seed", type=int, default=0, help="Global Random Seed")
parser.add_argument("--max_retries", type=int, default=1)
parser.add_argument("--text_guidance_scale", type=float, default=7.5)
parser.add_argument("--image_guidance_scale", type=float, default=2.5) # Higher for TAC logic
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/cub")

args = parser.parse_args()

# Environment
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
import openai
import clip

# [IMPORTS] Custom Modules
from taxonomy_aware_critic import taxonomy_aware_diagnosis # TAC Logic
from memory_guided_retrieval import retrieve_img_per_caption # MGR Logic
from global_memory import GlobalMemory # [Added] Global Memory Logic

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
    "classes_txt": "datasets/CUB_200_2011/classes.txt",
    "train_list": "datasets/CUB_200_2011/images.txt",
    "image_root": "datasets/CUB_200_2011/images",
    "output_path": "results/OmniGenV2_TAC_MGR_CUB"
}

# --- 4. Setup System ---
def setup_system():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(script_dir, args.omnigen2_path)))

    try:
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        pipe = OmniGen2Pipeline.from_pretrained(
            args.omnigen2_model_path,
            torch_dtype=torch.float16,
            transformer_lora_path=args.transformer_lora_path,
            trust_remote_code=True
        )
        # Patch for AttributeError
        if not hasattr(pipe.transformer, "enable_teacache"):
            pipe.transformer.enable_teacache = False
        pipe.to("cuda")
    except ImportError:
        print("Error: OmniGen2 not found.")
        sys.exit(1)

    # Clear Proxies for SiliconFlow
    # os.environ.pop("http_proxy", None)
    # os.environ.pop("https_proxy", None)
    # os.environ.pop("all_proxy", None)

    client = openai.OpenAI(
        api_key=args.openai_api_key,
        base_url="https://api.siliconflow.cn/v1/"
    )
    return pipe, client

def load_retrieval_db():
    print(f"Loading CUB DB...")
    paths = []
    with open(DATASET_CONFIG['train_list'], 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line: continue
            # CUB format: "1 001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg"
            image_filename = line.split(' ')[-1]
            img_path = os.path.join(DATASET_CONFIG['image_root'], image_filename)
            if os.path.exists(img_path):
                paths.append(img_path)
    print(f"Loaded {len(paths)} images.")
    return paths

def run_omnigen(pipe, prompt, input_images, output_path, seed, img_guidance_scale=None):
    # Ensure list format
    if isinstance(input_images, str):
        input_images = [input_images]

    processed_imgs = []
    for img in input_images:
        try:
            if isinstance(img, str): img = Image.open(img)
            if img.mode != 'RGB': img = img.convert('RGB')
            processed_imgs.append(img)
        except: continue

    # Deterministic Generator for this specific call
    generator = torch.Generator(device="cuda").manual_seed(seed)

    if img_guidance_scale is None:
        img_guidance_scale = args.image_guidance_scale

    pipe(
        prompt=prompt,
        input_images=processed_imgs,
        height=1024, width=1024,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=img_guidance_scale,
        num_inference_steps=50,
        generator=generator
    ).images[0].save(output_path)

# --- 5. Main Loop ---
if __name__ == "__main__":
    # 1. Set Seed
    seed_everything(args.seed)

    # 1.5 Pre-calculate Embeddings (Warmup)
    # Load DB first
    retrieval_db = load_retrieval_db()
    print("Pre-calculating/Loading retrieval embeddings on GPU...")
    try:
        # Run a dummy retrieval on the FULL database to force caching
        # We use a dummy query. The function will compute embeddings for all images in retrieval_db if not cached.
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

    # 2. Init
    pipe, client = setup_system()
    # retrieval_db = load_retrieval_db() # Already loaded
    os.makedirs(DATASET_CONFIG['output_path'], exist_ok=True)

    # [Added] Initialize Global Memory
    memory = GlobalMemory(
        memory_file="global_memory_cub.json",
        model_path="global_memory_cub.pth",
        device="cuda"
    )

    # 3. Load Tasks
    with open(DATASET_CONFIG['classes_txt'], 'r') as f:
        all_classes = [line.strip() for line in f.readlines() if line.strip()]

    my_classes = [c for i, c in enumerate(all_classes) if i % args.total_chunks == args.task_index]
    print(f"Processing {len(my_classes)} classes.")

    for class_name in tqdm(my_classes):
        # CUB Parsing: "1 001.Black_footed_Albatross" -> "Black footed Albatross"
        simple_name = class_name.split('.', 1)[-1].replace('_', ' ')
        safe_name = simple_name.replace(" ", "_").replace("/", "-")
        prompt = f"a photo of a {simple_name}"

        # Resume Logic
        final_success_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_FINAL.png")
        if os.path.exists(final_success_path):
            print(f"Skipping {safe_name}: Already finished.")
            continue

        log_file = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}.log")
        f_log = open(log_file, "w")
        f_log.write(f"Prompt: {prompt}\n")

        # Phase 1: Initial
        v1_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V1.png")

        # [Optimization] Shared Baseline Logic
        baseline_dir = "results/OmniGenV2_Baseline_CUB"
        baseline_v1_path = os.path.join(baseline_dir, f"{safe_name}_V1.png")

        if not os.path.exists(v1_path):
            if os.path.exists(baseline_v1_path):
                # print(f"Copying V1 from baseline: {baseline_v1_path}")
                shutil.copy(baseline_v1_path, v1_path)
            else:
                run_omnigen(pipe, prompt, [], v1_path, args.seed)
                # Try to populate baseline
                try:
                    os.makedirs(baseline_dir, exist_ok=True)
                    shutil.copy(v1_path, baseline_v1_path)
                except: pass

        current_image = v1_path
        retry_cnt = 0

        # [MGR Core]: Exclusion List to avoid bad references
        exclusion_list = []

        # [Score Tracking]
        best_score = -1
        best_image_path = None

        while retry_cnt < args.max_retries:
            f_log.write(f"\n--- Retry {retry_cnt+1} ---\n")

            # A. Taxonomy-Aware Critic
            diagnosis = taxonomy_aware_diagnosis(prompt, [current_image], client, args.llm_model)

            status = diagnosis.get('status')
            score = diagnosis.get('score', 0)
            error_type = diagnosis.get('error_type', 'other')
            features = diagnosis.get('features', [])
            critique = diagnosis.get('critique', '')

            f_log.write(f"Decision: {status} | Score: {score} | Type: {error_type}\nCritique: {critique}\n")

            # Update Best
            if score > best_score:
                best_score = score
                best_image_path = current_image

            if status == 'success':
                f_log.write(f">> Success! (Score: {score})\n")
                shutil.copy(current_image, final_success_path)
                break

            # B. Retrieval (MGR)
            # Query = Prompt + Features detected by TAC
            feature_text = ", ".join(features) if features else ""
            query = f"{prompt}. {feature_text}"
            # [Optimization] Use Hybrid Retrieval (CLIP + SigLIP) for better 1-shot performance
            # [Fix] Use device="cpu" to avoid OOM, relying on pre-calculated embeddings
            try:
                retrieved_lists, retrieved_scores = retrieve_img_per_caption(
                    [query], retrieval_db,
                    embeddings_path=args.embeddings_path,
                    k=50, device="cpu", method='Hybrid'
                )
                candidates = retrieved_lists[0]
                candidate_scores = retrieved_scores[0]
            except RuntimeError as e:
                f_log.write(f">> Retrieval Error (likely context length): {e}\n")
                # Fallback: use only prompt
                retrieved_lists, retrieved_scores = retrieve_img_per_caption(
                    [prompt], retrieval_db,
                    embeddings_path=args.embeddings_path,
                    k=50, device="cpu", method='Hybrid'
                )
                candidates = retrieved_lists[0]
                candidate_scores = retrieved_scores[0]

            # [Modified] Global Memory Re-ranking
            best_ref = None
            best_mem_score = -1.0

            # Check top candidates with Global Memory
            checked_count = 0
            for i, cand in enumerate(candidates):
                if cand in exclusion_list: continue

                # Predict score using the trained memory model
                mem_score = memory.predict_score(cand, query)

                if mem_score > best_mem_score:
                    best_mem_score = mem_score
                    best_ref = cand

                checked_count += 1
                if checked_count >= 10: break

            # Fallback
            if not best_ref:
                 for i, cand in enumerate(candidates):
                    if cand not in exclusion_list:
                        best_ref = cand
                        best_mem_score = min(1.0, candidate_scores[i] / 2.0)
                        break

            if best_ref:
                exclusion_list.append(best_ref)
            else:
                f_log.write(">> Memory Exhausted (No new unique references found).\n")
                break

            # [Adaptive Guidance Scale]
            # Heuristic: If similarity score is high (>0.3), trust the reference more (scale up to 3.5).
            # If low (<0.25), trust less (scale down to 2.0).
            # Base scale is 2.5.
            # Formula: scale = 1.0 + (score * 8.0) clamped to [2.0, 4.0]
            # Example: 0.25 * 8 = 2.0 + 1.0 = 3.0
            # Example: 0.30 * 8 = 2.4 + 1.0 = 3.4
            adaptive_scale = 2.0 + (best_mem_score * 3.0)
            adaptive_scale = max(2.5, min(adaptive_scale, 4.5))

            f_log.write(f">> Ref: {best_ref} (MemScore: {best_mem_score:.4f}) -> Adaptive Scale: {adaptive_scale:.2f}\n")

            # C. Dynamic Dispatch & Composition.4
            adaptive_scale = 2.0 + (best_ref_score * 5.0)
            adaptive_scale = max(2.5, min(adaptive_scale, 4.5))

            f_log.write(f">> Ref: {best_ref} (Score: {best_ref_score:.4f}) -> Adaptive Scale: {adaptive_scale:.2f}\n")
            f_log.write(f">> Ref: {best_ref} (Score: {best_ref_score:.4f}) -> Adaptive Scale: {adaptive_scale:.2f}\n")

            # C. Dynamic Dispatch & Composition
            next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")
        # Final Check for the last generated image if loop finished without success
        if not os.path.exists(final_success_path):
            f_log.write(f"\n--- Final Check (Last Generated) ---\n")
            # Evaluate the last image generated (current_image)
            if os.path.exists(current_image):
                diagnosis = taxonomy_aware_diagnosis(prompt, [current_image], client, args.llm_model)
                score = diagnosis.get('score', 0)
                f_log.write(f"Final Image Score: {score}\n")
                if score > best_score:
                    best_score = score
                    best_image_path = current_image

                # [Added] Record Feedback
                is_match = (diagnosis['status'] == 'success')
                correction = diagnosis.get('correction', None)
                if best_ref:
                    memory.add_feedback(
                        image_path=best_ref,
                        prompt=query,
                        actual_label=correction,
                        is_match=is_match
                    )

            f_log.write(f">> Loop finished. Best Score: {best_score}. Saving best image to FINAL.\n")
            if best_image_path and os.path.exists(best_image_path):
                shutil.copy(best_image_path, final_success_path)

        # [Added] Periodic Training
        if (my_classes.index(class_name) + 1) % 5 == 0:
            print(f"  >> Training Global Memory Model...")
            memory.train_model(epochs=5)

        f_log.close()mage = next_path
            retry_cnt += 1

        # Final Check for the last generated image if loop finished without success
        if not os.path.exists(final_success_path):
            f_log.write(f"\n--- Final Check (Last Generated) ---\n")
            # Evaluate the last image generated (current_image)
            if os.path.exists(current_image):
                diagnosis = taxonomy_aware_diagnosis(prompt, [current_image], client, args.llm_model)
                score = diagnosis.get('score', 0)
                f_log.write(f"Final Image Score: {score}\n")
                if score > best_score:
                    best_score = score
                    best_image_path = current_image

            f_log.write(f">> Loop finished. Best Score: {best_score}. Saving best image to FINAL.\n")
            if best_image_path and os.path.exists(best_image_path):
                shutil.copy(best_image_path, final_success_path)

        f_log.close()
