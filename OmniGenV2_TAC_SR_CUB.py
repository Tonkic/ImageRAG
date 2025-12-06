'''
OmniGenV2_TAC_SR_CUB.py
=============================
Configuration:
  - Generator: OmniGen V2
  - Critic: Taxonomy-Aware Critic (TAC) -> Fine-grained diagnosis
  - Retrieval: Static Retrieval (SR) -> No memory/exclusion list
  - Dataset: CUB-200-2011

Usage:
  python OmniGenV2_TAC_SR_CUB.py \
      --device_id 0 \
      --task_index 0 \
      --total_chunks 1 \
      --omnigen2_path ./OmniGen2 \
      --openai_api_key "sk-..."
'''

import argparse
import sys
import os

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="OmniGenV2 + TAC + SR (CUB)")

# Core Config
parser.add_argument("--device_id", type=int, required=True)
parser.add_argument("--task_index", type=int, default=0)
parser.add_argument("--total_chunks", type=int, default=1)

# Paths
parser.add_argument("--omnigen2_path", type=str, default="./OmniGen2")
parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2")
parser.add_argument("--transformer_lora_path", type=str, default="OmniGen2-EditScore7B" if os.path.exists("OmniGen2-EditScore7B") else None)
parser.add_argument("--openai_api_key", type=str, required=True)
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct") # SiliconFlow Default

# Params
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--text_guidance_scale", type=float, default=7.5)
parser.add_argument("--image_guidance_scale", type=float, default=2.5) # Higher guidance for composition
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/cub")

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
from static_retrieval import retrieve_img_per_caption      # The Static Retrieval logic

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
    "classes_txt": "datasets/CUB_200_2011/classes.txt",
    "train_list": "datasets/CUB_200_2011/images.txt",
    "image_root": "datasets/CUB_200_2011/images",
    "output_path": "results/OmniGenV2_TAC_SR_CUB"
}

# --- 3. Setup ---
def setup_system():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(script_dir, args.omnigen2_path)))

    try:
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        pipe = OmniGen2Pipeline.from_pretrained(
            args.omnigen2_model_path,
            torch_dtype=torch.float16,
            transformer_lora_path=args.transformer_lora_path,
            trust_remote_code=True, # Required for custom code
            mllm_kwargs={"attn_implementation": "flash_attention_2"}
        )
        # Patch for missing attribute
        if not hasattr(pipe.transformer, "enable_teacache"):
            pipe.transformer.enable_teacache = False
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.to("cuda")
    except ImportError:
        print("Error: OmniGen2 not found.")
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
    print("Loading CUB DB...")
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
    return paths

def run_omnigen(pipe, prompt, input_images, output_path, seed, img_guidance_scale=None):
    if isinstance(input_images, str): input_images = [input_images]

    processed = []
    for img in input_images:
        if isinstance(img, str): img = Image.open(img)
        if img.mode != 'RGB': img = img.convert('RGB')
        processed.append(img)

    gen = torch.Generator("cuda").manual_seed(seed)

    if img_guidance_scale is None:
        img_guidance_scale = args.image_guidance_scale

    pipe(
        prompt=prompt,
        input_images=processed,
        height=1024, width=1024,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=img_guidance_scale,
        num_inference_steps=50,
        generator=gen
    ).images[0].save(output_path)

# --- 4. Main ---
if __name__ == "__main__":
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
            device="cuda"
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
        # CUB Parsing: "1 001.Black_footed_Albatross" -> "Black footed Albatross"
        simple_name = class_name.split('.', 1)[-1].replace('_', ' ')
        safe_name = simple_name.replace(" ", "_").replace("/", "-")
        prompt = f"a photo of a {simple_name}"

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

        # [Score Tracking]
        best_score = -1
        best_image_path = None

        # Loop (Static Retrieval = No Exclusion List)
        while retry_cnt < args.max_retries:
            f_log.write(f"\n--- Retry {retry_cnt+1} ---\n")

            # A. Taxonomy-Aware Critic
            diagnosis = taxonomy_aware_diagnosis(prompt, [current_image], client, args.llm_model)
            status = diagnosis.get("status")
            score = diagnosis.get("score", 0)
            error_type = diagnosis.get("error_type", "other")
            needed_modifications = diagnosis.get('needed_modifications', [])
            correct_features = diagnosis.get('correct_features', [])
            critique = diagnosis.get('critique', '')

            f_log.write(f"Decision: {status} | Score: {score} | Type: {error_type}\nCritique: {critique}\n")
            f_log.write(f"Correct: {correct_features}\nMods: {needed_modifications}\n")

            # Update Best
            if score > best_score:
                best_score = score
                best_image_path = current_image

            if status == "success":
                f_log.write(f">> Success! (Score: {score})\n")
                shutil.copy(current_image, os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_FINAL.png"))
                break

            # B. Static Retrieval
            # Query = Prompt + Needed Modifications (Target State)
            mod_text = ", ".join(needed_modifications)

            query_parts = [prompt]
            if needed_modifications: query_parts.append(mod_text)
            query = ". ".join(query_parts)

            # [FIX] Truncate query to avoid CLIP context length overflow (77 tokens)
            if len(query) > 300:
                query = query[:300] + "..."

            f_log.write(f"Query: {query}\n")

            # Retrieve top-1 (No exclusion list check)
            # [Fix] Use device="cpu" to avoid OOM
            try:
                retrieved_lists, retrieved_scores = retrieve_img_per_caption(
                    [query], retrieval_db,
                    embeddings_path=args.embeddings_path,
                    k=1, device="cpu"
                )
            except RuntimeError as e:
                f_log.write(f">> Retrieval Error (likely context length): {e}\n")
                # Fallback: use only prompt
                retrieved_lists, retrieved_scores = retrieve_img_per_caption(
                    [prompt], retrieval_db,
                    embeddings_path=args.embeddings_path,
                    k=1, device="cpu"
                )

            if not retrieved_lists[0]:
                f_log.write(">> Retrieval failed.\n")
                break

            best_ref = retrieved_lists[0][0] # Always take Top-1
            best_ref_score = retrieved_scores[0][0]
            # [Adaptive Guidance Scale]
            # Optimization: Unified scale centered around 3.0
            # Formula: 2.0 + (score * 4.0). Range: [2.6, 3.4]
            adaptive_scale = 2.0 + (best_ref_score * 4.0)
            adaptive_scale = max(2.6, min(adaptive_scale, 3.4))

            # [Fix] Relax scale for wrong_concept to force change
            if error_type == "wrong_concept":
                 adaptive_scale = max(adaptive_scale, 3.0)

            f_log.write(f">> Static Ref: {best_ref} (Score: {best_ref_score:.4f}) -> Adaptive Scale: {adaptive_scale:.2f}\n")

            # C. Dynamic Dispatch (Based on Error Type)
            next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")

            # Construct correction instruction (Proactive & Concise)
            # "Fix [error_type]. [Modifications]."
            instruction_parts = []
            if needed_modifications:
                instruction_parts.append(f"{mod_text}")
            else:
                instruction_parts.append(f"Fix {error_type}")

            correction_instruction = ". ".join(instruction_parts)

            # Logic: Compositional errors -> Regen; Detail errors -> Edit
            if error_type in ["role_binding_error", "attribute_binding_error", "spatial_relation_error", "wrong_concept", "missing_object"]:
                # Regeneration Strategy
                regen_prompt = f"{prompt}. {correction_instruction}. Use <|image_1|> as a visual reference."
                f_log.write(f"Regen Prompt: {regen_prompt}\n")
                run_omnigen(pipe, regen_prompt, [best_ref], next_path, args.seed + retry_cnt + 1, img_guidance_scale=adaptive_scale)
            else:
                # Editing Strategy (e.g., text_error, style_error, count_error)
                edit_prompt = f"Edit this image to {correction_instruction}. Reference style: <|image_1|>"
                f_log.write(f"Edit Prompt: {edit_prompt}\n")
                edit_scale = max(2.5, adaptive_scale)
                run_omnigen(pipe, edit_prompt, [current_image, best_ref], next_path, args.seed + retry_cnt + 1, img_guidance_scale=edit_scale)

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
