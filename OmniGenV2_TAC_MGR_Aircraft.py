'''
OmniGenV2_TAC_MGR_Aircraft.py
=============================
Configuration:
  - Generator: OmniGen V2
  - Critic: Taxonomy-Aware Critic (TAC) -> Fine-grained diagnosis
  - Retrieval: Memory-Guided Retrieval (MGR) -> Dynamic RAG with Exclusion List
  - Dataset: FGVC-Aircraft

Usage:
  python OmniGenV2_TAC_MGR_Aircraft.py \
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
parser = argparse.ArgumentParser(description="OmniGenV2 + TAC + MGR (Aircraft)")

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
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/aircraft")

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
    "classes_txt": "datasets/fgvc-aircraft-2013b/data/variants.txt",
    "train_list": "datasets/fgvc-aircraft-2013b/data/images_train.txt",
    "image_root": "datasets/fgvc-aircraft-2013b/data/images",
    "output_path": "results/OmniGenV2_TAC_MGR_Aircraft"
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

    # 2. Load DB & Pre-calculate Embeddings (BEFORE loading OmniGen)
    # This prevents OOM by using GPU for retrieval caching while it's free.
    retrieval_db = load_retrieval_db()

    print("Pre-calculating/Loading retrieval embeddings on GPU...")
    try:
        # Run a dummy retrieval on the FULL database to force caching of all images
        # We use device="cuda" here because OmniGen isn't loaded yet.
        retrieve_img_per_caption(
            ["warmup_query"],
            retrieval_db,
            embeddings_path=args.embeddings_path,
            k=1,
            device="cuda",
            method='Hybrid'
        )
        # Clear GPU cache after retrieval model is done
        torch.cuda.empty_cache()
        print("Retrieval embeddings cached successfully.")
    except Exception as e:
        print(f"Warning during embedding caching: {e}")

    # 3. Init OmniGen (Now safe to load large model)
    pipe, client = setup_system()
    os.makedirs(DATASET_CONFIG['output_path'], exist_ok=True)

    # [Added] Initialize Global Memory
    memory = GlobalMemory(
        memory_file="global_memory_aircraft.json",
        model_path="global_memory_aircraft.pth",
        device="cuda"
    )

    # 4. Load Tasks
    with open(DATASET_CONFIG['classes_txt'], 'r') as f:
        all_classes = [line.strip() for line in f.readlines() if line.strip()]

    my_classes = [c for i, c in enumerate(all_classes) if i % args.total_chunks == args.task_index]
    print(f"Processing {len(my_classes)} classes.")

    for class_name in tqdm(my_classes):
        safe_name = class_name.replace(" ", "_").replace("/", "-")
        prompt = f"a photo of a {class_name}"

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
        baseline_dir = "results/OmniGenV2_Baseline_Aircraft"
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
            # B. Memory-Guided Retrieval
            # [Optimization] Use Hybrid Retrieval (CLIP + SigLIP) for better 1-shot performance
            # [Fix] Use device="cpu" to avoid OOM. Embeddings are already cached on disk from the pre-calculation step.
            # Encoding just the text query on CPU is fast enough.
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
            # We check top 10 candidates that are not in exclusion list
            checked_count = 0
            for i, cand in enumerate(candidates):
                if cand in exclusion_list: continue

                # Predict score using the trained memory model
                mem_score = memory.predict_score(cand, query)

                if mem_score > best_mem_score:
                    best_mem_score = mem_score
                    best_ref = cand

                checked_count += 1
                if checked_count >= 10: break # Only re-rank top 10 valid candidates

            # Fallback: if memory didn't find anything good (or model is untrained), use the top retrieval result
            if not best_ref:
                 for i, cand in enumerate(candidates):
                    if cand not in exclusion_list:
                        best_ref = cand
                        # Normalize retrieval score roughly to 0-1 range for guidance scale logic
                        # Hybrid scores are around 0.03-0.06, scaled by 30 -> 0.9-1.8
                        # We clamp to 0-1 for consistency with memory score
                        best_mem_score = min(1.0, candidate_scores[i] / 2.0)
                        break

            if best_ref:
                exclusion_list.append(best_ref)
            else:
                f_log.write(">> Memory Exhausted (No new unique references found).\n")
                break

            # [Adaptive Guidance Scale]
            # Optimization: For fine-grained tasks (Aircraft), we need STRONG visual guidance.
            # Low scale (<2.5) causes hallucinations.
            # New Formula: Base 2.0 + (score * 5.0). Range: [2.5, 4.5]
            # We use the memory score (0-1) to drive the scale
            adaptive_scale = 2.0 + (best_mem_score * 3.0)
            adaptive_scale = max(2.5, min(adaptive_scale, 4.5))

            f_log.write(f">> Ref: {best_ref} (MemScore: {best_mem_score:.4f}) -> Adaptive Scale: {adaptive_scale:.2f}\n")

            # C. Dynamic Dispatch & Compositiontions.
            # New Formula: Base 2.0 + (score * 5.0). Range: [2.5, 4.5]
            adaptive_scale = 2.0 + (best_ref_score * 5.0)
            adaptive_scale = max(2.5, min(adaptive_scale, 4.5))
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

                # [Added] Record Feedback for the last attempt
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
        # Train every 5 tasks to keep the memory updated
        if (my_classes.index(class_name) + 1) % 5 == 0:
            print(f"  >> Training Global Memory Model...")
            memory.train_model(epochs=5)

        f_log.close().write(f"Edit Prompt: {edit_prompt}\n")
                # For Edit, we usually want stronger guidance, so we boost the adaptive scale slightly
                edit_scale = max(2.5, adaptive_scale)
                run_omnigen(pipe, edit_prompt, [current_image, best_ref], next_path, args.seed + retry_cnt + 1, img_guidance_scale=edit_scale)

            current_image = next_path
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