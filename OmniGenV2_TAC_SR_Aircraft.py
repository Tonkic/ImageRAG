'''
OmniGenV2_TAC_SR_Aircraft.py
=============================
Configuration:
  - Generator: OmniGen V2
  - Critic: Taxonomy-Aware Critic (TAC) -> Fine-grained diagnosis
  - Retrieval: Static Retrieval (SR) -> No memory/exclusion list
  - Dataset: FGVC-Aircraft

Usage:
  python OmniGenV2_TAC_SR_Aircraft.py \
      --device_id 0 \
      --task_index 0 \
      --total_chunks 1 \
      --omnigen2_path ./OmniGen2 \
      --openai_api_key "sk-..."
'''

import argparse
import sys
import os
# [Proxy Config] Clear system proxies for direct connection
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="OmniGenV2 + TAC + SR (Aircraft)")

# Core Config
parser.add_argument("--device_id", type=int, required=True)
parser.add_argument("--task_index", type=int, default=0)
parser.add_argument("--total_chunks", type=int, default=1)

# Paths
parser.add_argument("--omnigen2_path", type=str, default="./OmniGen2")
parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2")
parser.add_argument("--transformer_lora_path", type=str, default=None)
parser.add_argument("--openai_api_key", type=str, required=True)
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct") # SiliconFlow Default

# Params
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--text_guidance_scale", type=float, default=7.5)
parser.add_argument("--image_guidance_scale", type=float, default=1.5) # Higher guidance for composition
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/aircraft")
parser.add_argument("--retrieval_method", type=str, default="Hybrid", choices=["CLIP", "Hybrid", "ColPali"])

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
print(f"DEBUG: CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")

import json
import shutil
import numpy as np
import torch
print(f"DEBUG: Torch sees {torch.cuda.device_count()} devices. Current device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})")
import random
from PIL import Image
from tqdm import tqdm
import openai
import clip

# [IMPORTS]
from taxonomy_aware_critic import taxonomy_aware_diagnosis # The new Critic
from memory_guided_retrieval import retrieve_img_per_caption      # The Static Retrieval logic

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
    "output_path": "results/OmniGenV2_TAC_SR_Aircraft"
}

# --- 3. Setup ---
def setup_system():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(script_dir, args.omnigen2_path)))
    try:
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        pipe = OmniGen2Pipeline.from_pretrained(
            args.omnigen2_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True, # Required for custom code
        )
        # Patch for missing attribute
        if not hasattr(pipe.transformer, "enable_teacache"):
            pipe.transformer.enable_teacache = False
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.to("cuda")
    except ImportError as e:
        print(f"Error: OmniGen2 not found. Details: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # os.environ.pop("http_proxy", None)
    # os.environ.pop("https_proxy", None)
    # os.environ.pop("all_proxy", None)

    print("Initializing OpenAI Client for SiliconFlow...")
    client = openai.OpenAI(
        api_key=args.openai_api_key,
        base_url="https://api.siliconflow.cn/v1/"
    )
    return pipe, client

def load_db():
    print("Loading DB...")
    paths = []
    with open(DATASET_CONFIG['train_list'], 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line: continue
            img_path = os.path.join(DATASET_CONFIG['image_root'], f"{line}.jpg")
            if os.path.exists(img_path):
                paths.append(img_path)
    return paths

def run_omnigen(pipe, prompt, input_images, output_path, seed, img_guidance_scale=None, text_guidance_scale=None):
    if isinstance(input_images, str): input_images = [input_images]

    processed = []
    for img in input_images:
        if isinstance(img, str): img = Image.open(img)
        if img.mode != 'RGB': img = img.convert('RGB')
        processed.append(img)

    gen = torch.Generator("cuda").manual_seed(seed)

    if img_guidance_scale is None:
        img_guidance_scale = args.image_guidance_scale

    if text_guidance_scale is None:
        text_guidance_scale = args.text_guidance_scale

    pipe(
        prompt=prompt,
        input_images=processed,
        height=1024, width=1024,
        text_guidance_scale=text_guidance_scale,
        image_guidance_scale=img_guidance_scale,
        num_inference_steps=50,
        generator=gen
    ).images[0].save(output_path)

# --- 4. Main ---
if __name__ == "__main__":
    import time
    start_time = time.time()

    seed_everything(args.seed)

    # 1. Load DB & Pre-calculate Embeddings (Warmup)
    retrieval_db = load_db()
    print("Pre-calculating/Loading retrieval embeddings on GPU...")
    try:
        # Run a dummy retrieval on the FULL database to force caching
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
    # retrieval_db = load_db() # Already loaded
    os.makedirs(DATASET_CONFIG['output_path'], exist_ok=True)

    with open(DATASET_CONFIG['classes_txt']) as f:
        all_classes = [l.strip() for l in f.readlines() if l.strip()]

    my_tasks = [c for i, c in enumerate(all_classes) if i % args.total_chunks == args.task_index]
    print(f"Processing {len(my_tasks)} classes.")

    for class_name in tqdm(my_tasks):
        safe_name = class_name.replace(" ", "_").replace("/", "-")
        prompt = f"a photo of a {class_name}"

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

        # [Score Tracking]
        best_score = -1
        best_image_path = None

        # Loop (Static Retrieval = No Exclusion List)
        while retry_cnt < args.max_retries:
            f_log.write(f"\n--- Retry {retry_cnt+1} ---\n")

            # A. TAC Diagnosis (V4.0)
            diagnosis = taxonomy_aware_diagnosis(prompt, [current_image], client, args.llm_model)

            score = diagnosis.get('final_score', 0)
            taxonomy_status = diagnosis.get('taxonomy_check', 'unknown')
            critique = diagnosis.get('critique', '')
            refined_prompt = diagnosis.get('refined_prompt', prompt)

            f_log.write(f"Decision: Score {score} | Taxonomy: {taxonomy_status}\nCritique: {critique}\n")

            # Update Best
            if score > best_score:
                best_score = score
                best_image_path = current_image

            if score >= 8.0:
                f_log.write(f">> Success! (Score: {score})\n")
                shutil.copy(current_image, os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_FINAL.png"))
                break

            # B. Static Retrieval (Simplified for SR)
            query = refined_prompt
            if len(query) > 300: query = query[:300]

            # [Token Length Check]
            from memory_guided_retrieval import check_token_length
            check_token_length([query], device="cpu", method=args.retrieval_method)

            try:
                retrieved_lists, retrieved_scores = retrieve_img_per_caption(
                    [query], retrieval_db,
                    embeddings_path=args.embeddings_path,
                    k=1, device="cuda", method=args.retrieval_method
                )
                best_ref = retrieved_lists[0][0]
                best_ref_score = retrieved_scores[0][0]
            except Exception as e:
                f_log.write(f">> Retrieval Error: {e}\n")
                # Fallback
                retrieved_lists, retrieved_scores = retrieve_img_per_caption(
                    [prompt], retrieval_db,
                    embeddings_path=args.embeddings_path,
                    k=1, device="cuda", method=args.retrieval_method
                )
                best_ref = retrieved_lists[0][0]
                best_ref_score = retrieved_scores[0][0]

            f_log.write(f">> Static Ref: {best_ref} (Score: {best_ref_score:.4f})\n")

            # C. Generation Strategy
            next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")

            # [Modified] Unified Guidance Scale (Same as BC)
            current_img_guidance = 1.5 # Fixed value as requested
            current_text_guidance = args.text_guidance_scale
            f_log.write(f">> Strategy: Unified Guidance (Image: {current_img_guidance}, Text: {current_text_guidance})\n")

            # User Request: Use original_prompt + visual_keywords (refined_prompt)
            if refined_prompt != prompt:
                gen_prompt = f"{prompt}. {refined_prompt}. Use reference image <|image_1|>."
            else:
                gen_prompt = f"{prompt}. Use reference image <|image_1|>."

            run_omnigen(pipe, gen_prompt, [best_ref], next_path, args.seed + retry_cnt + 1, img_guidance_scale=current_img_guidance, text_guidance_scale=current_text_guidance)

            current_image = next_path
            retry_cnt += 1

        # Final Check
        final_success_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_FINAL.png")
        if not os.path.exists(final_success_path):
            f_log.write(f"\n--- Final Check (Last Generated) ---\n")
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

    end_time = time.time()
    elapsed_time = end_time - start_time
    with open(os.path.join(DATASET_CONFIG['output_path'], "time_elapsed.txt"), "w") as f:
        f.write(f"Total execution time: {elapsed_time:.2f} seconds\n")