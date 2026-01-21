'''
OmniGenV2_TAC_exp_noRAG_Aircraft.py
=============================
Configuration:
  - Generator: OmniGen V2
  - Critic: Taxonomy-Aware Critic (TAC) -> Fine-grained diagnosis
  - Retrieval: None (noRAG) -> Prompt Refinement only
  - Policy: Training-Free GRPO (Group Relative Policy Optimization) with Experience Library
  - Dataset: FGVC-Aircraft

Usage:
  python OmniGenV2_TAC_exp_noRAG_Aircraft.py \
      --device_id 0 \
      --task_index 0 \
      --total_chunks 1 \
      --omnigen2_path ./OmniGen2 \
      --openai_api_key "sk-..."
'''
from datetime import datetime


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
parser = argparse.ArgumentParser(description="OmniGenV2 + TAC + exp + noRAG (Aircraft)")

# Core Config
parser.add_argument("--device_id", type=str, required=True, help="Main device ID (e.g. '0' or '0,1')")
parser.add_argument("--vlm_device_id", type=str, default=None, help="Device ID for VLM (if different)")
parser.add_argument("--task_index", type=int, default=0)
parser.add_argument("--total_chunks", type=int, default=1)

# Paths
parser.add_argument("--omnigen2_path", type=str, default="./OmniGen2")
parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2")
parser.add_argument("--transformer_lora_path", type=str, default=None)
parser.add_argument("--openai_api_key", type=str, required=False, help="Required for SiliconFlow API. If not provided, uses local model weights.")
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct") # SiliconFlow Default

# Local Weights Config
parser.add_argument("--use_local_model_weight", action="store_true", help="Load local model weights directly (transformers)")
parser.add_argument("--local_model_weight_path", type=str, default="/home/tingyu/imageRAG/Qwen3-VL-4B-Instruct")
parser.add_argument("--enable_offload", action="store_true", help="Enable CPU offloading for OmniGen to save VRAM")

# Params
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--text_guidance_scale", type=float, default=7.5)
parser.add_argument("--image_guidance_scale", type=float, default=1.5) # Higher guidance for composition
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/aircraft") # Not used
parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=["CLIP", "LongCLIP", "SigLIP", "SigLIP2", "Qwen2.5-VL", "Qwen3-VL"], help="Retrieval Model")
parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter (for Qwen2.5-VL)")
parser.add_argument("--group_size", type=int, default=4, help="Group size for GRPO")

args = parser.parse_args()

# Handle Devices
if args.vlm_device_id:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device_id},{args.vlm_device_id}"
    omnigen_device = "cuda:0"
    vlm_device_map = {"": "cuda:1"}
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    omnigen_device = "cuda:0"
    vlm_device_map = "auto"

print(f"DEBUG: CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")

import json
import shutil
import numpy as np
import torch
import random
from PIL import Image
from tqdm import tqdm
import openai
import clip
import time
import base64
import io
import psutil
import threading
import matplotlib.pyplot as plt
import datetime

print(f"DEBUG: OmniGen Device: {omnigen_device}, VLM Device Map: {vlm_device_map}")
print(f"DEBUG: Torch sees {torch.cuda.device_count()} devices.")

# [IMPORTS]
from taxonomy_aware_critic import (
    taxonomy_aware_diagnosis, # The new Critic
    rate_image_match,
    extract_semantic_advantage,
    message_gpt
)
from rag_utils import (
    ResourceMonitor,
    UsageTrackingClient,
    ExperienceLibrary,
    LocalQwen3VLWrapper,
    seed_everything,
    RUN_STATS
)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 2. Config ---

dt = datetime.now()
timestamp = dt.strftime("%Y.%-m.%-d")
run_time = dt.strftime("%H-%M-%S")
try:
    _rm = args.retrieval_method
except:
    _rm = "default"

DATASET_CONFIG = {
    "classes_txt": "datasets/fgvc-aircraft-2013b/data/variants.txt",
    "train_list": "datasets/fgvc-aircraft-2013b/data/images_train.txt",
    "image_root": "datasets/fgvc-aircraft-2013b/data/images",
    "output_path": f"results/{_rm}/{timestamp}/OmniGenV2_TAC_exp_noRAG_Aircraft_{run_time}"
}

# --- 3. Setup ---
# ResourceMonitor is imported from rag_utils

def setup_system(omnigen_device, vlm_device_map):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(script_dir, args.omnigen2_path)))
    try:
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        pipe = OmniGen2Pipeline.from_pretrained(
            args.omnigen2_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True, # Required for custom code
        )
        if not hasattr(pipe.transformer, "enable_teacache"):
            pipe.transformer.enable_teacache = False
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        if args.enable_offload:
            print("Enabling model CPU offload for OmniGen...")
            pipe.enable_model_cpu_offload(device=omnigen_device)
        else:
            pipe.to(omnigen_device)
    except ImportError as e:
        print(f"Error: OmniGen2 not found. Details: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("Initializing Client...")
    # Logic: Missing API Key -> Use Local Weights
    if not args.openai_api_key:
        print(f"  Using Local Model Weights from {args.local_model_weight_path}")
        client = LocalQwen3VLWrapper(args.local_model_weight_path, device_map=vlm_device_map)
        # Override llm_model arg to avoid confusion, though wrapper ignores it
        args.llm_model = "local-qwen3-vl"
    else:
        print("  Using SiliconFlow API...")
        client = openai.OpenAI(
            api_key=args.openai_api_key,
            base_url="https://api.siliconflow.cn/v1/"
        )

    # Wrap client for usage tracking
    client = UsageTrackingClient(client)
    return pipe, client

def run_omnigen(pipe, prompt, input_images, output_path, seed, img_guidance_scale=1.5, text_guidance_scale=7.5):
    if isinstance(input_images, str):
        input_images = [input_images]

    processed = []
    for img in input_images:
        try:
            if isinstance(img, str): img = Image.open(img)
            if img.mode != 'RGB': img = img.convert('RGB')
            processed.append(img)
        except Exception as e:
            print(f"Error loading image: {e}")
            continue

    gen = torch.Generator(device=pipe.device).manual_seed(seed)

    pipe(
        prompt=prompt,
        input_images=processed,
        height=512, width=512,
        text_guidance_scale=text_guidance_scale,
        image_guidance_scale=img_guidance_scale,
        num_inference_steps=50,
        generator=gen
    ).images[0].save(output_path)

# --- 4. Main ---
if __name__ == "__main__":
    start_time = time.time()

    seed_everything(args.seed)

    pipe, client = setup_system(omnigen_device, vlm_device_map)
    os.makedirs(DATASET_CONFIG['output_path'], exist_ok=True)

    # Create logs directory
    logs_dir = os.path.join(DATASET_CONFIG['output_path'], "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Start Resource Monitor
    monitor = ResourceMonitor(interval=1.0)
    monitor.start()

    # Save Run Configuration
    config_path = os.path.join(logs_dir, "run_config.txt")
    with open(config_path, "w") as f:
        f.write("Run Configuration:\n")
        f.write("==================\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write(f"\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")

    with open(DATASET_CONFIG['classes_txt']) as f:
        all_classes = [l.strip() for l in f.readlines() if l.strip()]

    my_tasks = [c for i, c in enumerate(all_classes) if i % args.total_chunks == args.task_index]
    print(f"Processing {len(my_tasks)} classes.")

    # Initialize Experience Library
    experience_lib = ExperienceLibrary()

    for class_name in tqdm(my_tasks):
        # [Reset Specific Rules]
        experience_lib.reset_specific()

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
                shutil.copy(baseline_v1_path, v1_path)
            else:
                run_omnigen(pipe, prompt, [], v1_path, args.seed)
                try:
                    os.makedirs(baseline_dir, exist_ok=True)
                    shutil.copy(v1_path, baseline_v1_path)
                except: pass

        current_image = v1_path
        retry_cnt = 0

        # [Score Tracking]
        best_score = -1
        best_image_path = None

        # [Knowledge Retrieval] - Sanity Check
        from taxonomy_aware_critic import generate_knowledge_specs
        try:
            reference_specs = generate_knowledge_specs(class_name, client, args.llm_model)
            f_log.write(f"Reference Specs: {reference_specs}\n")
        except Exception as e:
            f_log.write(f"Reference Specs Retrieval Failed: {e}\n")
            reference_specs = None

        # Loop (Group Rollout)
        while retry_cnt < args.max_retries:
            f_log.write(f"\n--- Retry {retry_cnt+1} (Group Rollout) ---\n")

            # A. Diagnosis of current best (to get critique)
            diagnosis = taxonomy_aware_diagnosis(prompt, [current_image], client, args.llm_model, reference_specs=reference_specs)
            score = diagnosis.get('final_score', 0)
            taxonomy_status = diagnosis.get('taxonomy_check', 'unknown')
            critique = diagnosis.get('critique', '')

            f_log.write(f"Current Best Score: {score} | Taxonomy: {taxonomy_status}\nCritique: {critique}\n")

            if score > best_score:
                best_score = score
                best_image_path = current_image

            if score >= 8.0 or (score >= 6.0 and taxonomy_status == 'correct'):
                f_log.write(f">> Success! (Score: {score}, Taxonomy: {taxonomy_status})\n")
                shutil.copy(current_image, os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_FINAL.png"))
                break

            # B. Group Rollout
            group_results = []
            context_str = experience_lib.get_context_str()

            # Base instruction for PROMPT generation (noRAG)
            instruction = f"The current image of {class_name} has issues: {critique}. \n" \
                          f"We need to refine the prompt to fix these issues. \n" \
                          f"Refine the generation prompt for {class_name}. \n" \
                          f"{context_str}\n" \
                          f"Output ONLY the refined prompt string."

            f_log.write(f"Experience Context: {context_str}\n")

            for i in range(args.group_size):
                f_log.write(f"  > Group Member {i+1}/{args.group_size}...\n")

                # a. Generate Prompt Variation
                # Use high temperature for diversity
                variation_prompt = message_gpt(instruction, client, model=args.llm_model, temperature=0.9)
                if not variation_prompt: variation_prompt = prompt

                # Clean up prompt
                variation_prompt = variation_prompt.replace('"', '').replace("'", "").strip()
                f_log.write(f"    Variation Prompt: {variation_prompt}\n")

                # b. Generate Image (Seed Offset)
                trial_seed = args.seed + retry_cnt * 100 + i # Ensure unique seeds across retries and groups

                # [DEBUG] Log Seed
                print(f"DEBUG: Generating Group Member {i} | Seed: {trial_seed} | Prompt: {variation_prompt[:50]}...")
                f_log.write(f"DEBUG: Group {i} Seed: {trial_seed}\n")

                trial_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_retry{retry_cnt+1}_group{i}.png")

                run_omnigen(pipe, variation_prompt, [], trial_path, trial_seed,
                            img_guidance_scale=1.5, text_guidance_scale=args.text_guidance_scale)

                # c. Evaluate
                trial_diagnosis = taxonomy_aware_diagnosis(prompt, [trial_path], client, args.llm_model, reference_specs=reference_specs)
                trial_score = trial_diagnosis.get('final_score', 0)
                trial_critique = trial_diagnosis.get('critique', '')

                group_results.append({
                    "path": trial_path,
                    "score": trial_score,
                    "prompt": variation_prompt, # Store prompt instead of query
                    "critique": trial_critique
                })
                f_log.write(f"    > Score: {trial_score}\n")

            # C. Selection & Update
            # Sort by score
            group_results.sort(key=lambda x: x['score'], reverse=True)
            best_in_group = group_results[0]

            f_log.write(f">> Best in Group: {best_in_group['path']} (Score: {best_in_group['score']})\n")

            # Update Current Image
            current_image = best_in_group['path']

            # Update Experience Library
            # Learn from the group (Semantic Advantage)
            new_rule_response = extract_semantic_advantage(group_results, client, args.llm_model)
            if new_rule_response:
                if "[GLOBAL]" in new_rule_response:
                    rule = new_rule_response.replace("[GLOBAL]", "").strip()
                    experience_lib.add_rule(rule, is_global=True)
                    f_log.write(f">> Learned New GLOBAL Rule: {rule}\n")
                    print(f"Learned New GLOBAL Rule: {rule}")
                elif "[SPECIFIC]" in new_rule_response:
                    rule = new_rule_response.replace("[SPECIFIC]", "").strip()
                    experience_lib.add_rule(rule, is_global=False)
                    f_log.write(f">> Learned New SPECIFIC Rule: {rule}\n")
                    print(f"Learned New SPECIFIC Rule: {rule}")
                else:
                    # Default to specific if no tag
                    experience_lib.add_rule(new_rule_response, is_global=False)
                    f_log.write(f">> Learned New Rule (Default Specific): {new_rule_response}\n")
                    print(f"Learned New Rule: {new_rule_response}")

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

    monitor.stop()
    monitor.save_plots(logs_dir)

    end_time = time.time()
    elapsed_time = end_time - start_time
    with open(os.path.join(logs_dir, "time_elapsed.txt"), "w") as f:
        f.write(f"Total execution time: {elapsed_time:.2f} seconds\n")
        f.write(f"Total Input Tokens: {RUN_STATS['input_tokens']}\n")
        f.write(f"Total Output Tokens: {RUN_STATS['output_tokens']}\n")
        f.write(f"Total Tokens: {RUN_STATS['input_tokens'] + RUN_STATS['output_tokens']}\n")
