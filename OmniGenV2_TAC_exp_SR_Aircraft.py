'''
OmniGenV2_TAC_exp_SR_Aircraft.py
=============================
Configuration:
  - Generator: OmniGen V2
  - Critic: Taxonomy-Aware Critic (TAC) -> Fine-grained diagnosis
  - Retrieval: Static Retrieval (SR) -> No memory/exclusion list
  - Policy: Training-Free GRPO (Group Relative Policy Optimization) with Experience Library
  - Dataset: FGVC-Aircraft

Usage:
  python OmniGenV2_TAC_exp_SR_Aircraft.py \
      --device_id 0 \
      --task_index 0 \
      --total_chunks 1 \
      --omnigen2_path ./OmniGen2 \
      --openai_api_key "sk-..."
'''

import argparse
import sys
import os
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

# [Proxy Config] Clear system proxies for direct connection
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="OmniGenV2 + TAC + exp + SR (Aircraft)")

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

# Params
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--text_guidance_scale", type=float, default=7.5)
parser.add_argument("--image_guidance_scale", type=float, default=1.5) # Higher guidance for composition
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/aircraft")
parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=["CLIP", "LongCLIP", "SigLIP", "SigLIP2", "Qwen2.5-VL", "Qwen3-VL"], help="Retrieval Model")
parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter (for Qwen2.5-VL)")
parser.add_argument("--group_size", type=int, default=4, help="Group size for GRPO")
parser.add_argument("--retrieval_datasets", nargs='+', default=['aircraft'], choices=['aircraft', 'cub', 'imagenet'], help="Datasets to use for retrieval")

args = parser.parse_args()

# Handle Devices
if args.vlm_device_id:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device_id},{args.vlm_device_id}"
    omnigen_device = "cuda:0"
    vlm_device_map = {"": "cuda:1"}
    retrieval_device = "cuda:1" # Offload retrieval to VLM GPU
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    omnigen_device = "cuda:0"
    vlm_device_map = "auto"
    retrieval_device = "cuda:0"

print(f"DEBUG: CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"DEBUG: OmniGen Device: {omnigen_device}, VLM Device Map: {vlm_device_map}, Retrieval Device: {retrieval_device}")
print(f"DEBUG: Torch sees {torch.cuda.device_count()} devices.")

# [IMPORTS]
from taxonomy_aware_critic import (
    taxonomy_aware_diagnosis, # The new Critic
    rate_image_match,
    extract_semantic_advantage,
    message_gpt,
    encode_image
)
from memory_guided_retrieval import retrieve_img_per_caption      # The Static Retrieval logic
from rag_utils import (
    ResourceMonitor,
    UsageTrackingClient,
    ExperienceLibrary,
    LocalQwen3VLWrapper,
    seed_everything,
    RUN_STATS
)

# --- Global Stats & Monitoring ---
# RUN_STATS is imported from rag_utils

# --- 2. Config ---
DATASET_CONFIG = {
    "classes_txt": "datasets/fgvc-aircraft-2013b/data/variants.txt",
    "train_list": "datasets/fgvc-aircraft-2013b/data/images_train.txt",
    "image_root": "datasets/fgvc-aircraft-2013b/data/images",
    "output_path": "results/OmniGenV2_TAC_exp_SR_Aircraft"
}

# --- Experience Library ---
# ExperienceLibrary is imported from rag_utils

# --- Local Qwen3-VL Wrapper ---
# LocalQwen3VLWrapper is imported from rag_utils

# --- 3. Setup ---
def setup_system(omnigen_device, vlm_device_map):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(script_dir, args.omnigen2_path)))
    try:
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        pipe = OmniGen2Pipeline.from_pretrained(
            args.omnigen2_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if not hasattr(pipe.transformer, "enable_teacache"):
            pipe.transformer.enable_teacache = False
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.to(omnigen_device)
    except ImportError as e:
        print(f"Error: OmniGen2 not found. Details: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("Initializing Client...")
    # Logic: Explicit Local Flag OR Missing API Key -> Use Local Weights
    if args.use_local_model_weight or not args.openai_api_key:
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

def load_db():
    print(f"Loading Retrieval DBs: {args.retrieval_datasets}...")
    all_paths = []

    for ds in args.retrieval_datasets:
        if ds == 'aircraft':
            print("  Loading Aircraft...")
            root = "datasets/fgvc-aircraft-2013b/data/images"
            list_file = "datasets/fgvc-aircraft-2013b/data/images_train.txt"
            if os.path.exists(list_file):
                with open(list_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        path = os.path.join(root, f"{line}.jpg")
                        if os.path.exists(path):
                            all_paths.append(path)
            else:
                print(f"  Warning: Aircraft list file not found at {list_file}")

        elif ds == 'cub':
            print("  Loading CUB...")
            root = "datasets/CUB_200_2011/images"
            split_file = "datasets/CUB_200_2011/train_test_split.txt"
            images_file = "datasets/CUB_200_2011/images.txt"

            if os.path.exists(split_file) and os.path.exists(images_file):
                train_ids = set()
                with open(split_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2 and parts[1] == '1':
                            train_ids.add(parts[0])

                with open(images_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            img_id = parts[0]
                            rel_path = parts[1]
                            if img_id in train_ids:
                                path = os.path.join(root, rel_path)
                                if os.path.exists(path):
                                    all_paths.append(path)
            else:
                print(f"  Warning: CUB files not found at {split_file} or {images_file}")

        elif ds == 'imagenet':
            print("  Loading ImageNet...")
            root = "/home/tingyu/imageRAG/datasets/ILSVRC2012_train"
            list_file = "/home/tingyu/imageRAG/datasets/imagenet_train_list.txt"
            if os.path.exists(list_file):
                with open(list_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        path = os.path.join(root, line)
                        if os.path.exists(path):
                            all_paths.append(path)
            else:
                print(f"  Warning: ImageNet list file not found at {list_file}")

    print(f"Total loaded retrieval images: {len(all_paths)}")
    return all_paths

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

    # 1. Load DB & Pre-calculate Embeddings (Warmup)
    retrieval_db = load_db()
    print("Pre-calculating/Loading retrieval embeddings on GPU...")
    try:
        retrieve_img_per_caption(
            ["warmup_query"],
            retrieval_db,
            embeddings_path=args.embeddings_path,
            k=1,
            device=retrieval_device,
            method=args.retrieval_method
        )
        torch.cuda.empty_cache()
        print("Retrieval embeddings cached successfully.")
    except Exception as e:
        print(f"Warning during embedding caching: {e}")

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

            # Base instruction for query generation
            instruction = f"The current image of {class_name} has issues: {critique}. \n" \
                          f"We need to retrieve a better reference image to fix these issues. \n" \
                          f"Refine the retrieval query for {class_name}. \n" \
                          f"{context_str}\n" \
                          f"Output ONLY the retrieval query string."

            f_log.write(f"Experience Context: {context_str}\n")

            for i in range(args.group_size):
                f_log.write(f"  > Group Member {i+1}/{args.group_size}...\n")

                # a. Generate Query Variation
                # Use high temperature for diversity
                variation_query = message_gpt(instruction, client, model=args.llm_model, temperature=0.9)
                if not variation_query: variation_query = f"{class_name} {class_name}"

                # Clean up query (remove quotes etc if any)
                variation_query = variation_query.replace('"', '').replace("'", "").strip()

                # b. Retrieve
                try:
                    retrieved_lists, retrieved_scores = retrieve_img_per_caption(
                        [variation_query], retrieval_db,
                        embeddings_path=args.embeddings_path,
                        k=3, device=retrieval_device, method=args.retrieval_method,
                        adapter_path=args.adapter_path
                    )
                    candidates = retrieved_lists[0]

                    best_candidate = None
                    best_confidence = -1

                    # [Modified] Best-of-N Selection
                    for img_path in candidates:
                        score = rate_image_match(img_path, class_name, reference_specs, client, args.llm_model)
                        if score > best_confidence:
                            best_confidence = score
                            best_candidate = img_path

                    if best_confidence >= 3:
                        ref_img = best_candidate
                        f_log.write(f"    >> Selected ref {os.path.basename(ref_img)} with confidence {best_confidence}/10.\n")
                    else:
                        ref_img = None
                        f_log.write(f"    >> All refs rejected (Best Score: {best_confidence}). Fallback to Text-Only.\n")

                except Exception as e:
                    f_log.write(f"    Retrieval failed: {e}. Using random.\n")
                    if retrieval_db: ref_img = random.choice(retrieval_db)
                    else: continue

                # c. Generate Image (Seed Offset)
                trial_seed = args.seed + retry_cnt * 100 + i # Ensure unique seeds across retries and groups

                # [DEBUG] Log Seed
                print(f"DEBUG: Generating Group Member {i} | Seed: {trial_seed} | Query: {variation_query[:50]}...")
                f_log.write(f"DEBUG: Group {i} Seed: {trial_seed}\n")

                trial_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_retry{retry_cnt+1}_group{i}.png")

                imgs_input = [ref_img] if ref_img else []
                run_omnigen(pipe, prompt, imgs_input, trial_path, trial_seed,
                            img_guidance_scale=1.5, text_guidance_scale=args.text_guidance_scale)

                # d. Evaluate
                trial_diagnosis = taxonomy_aware_diagnosis(prompt, [trial_path], client, args.llm_model, reference_specs=reference_specs)
                trial_score = trial_diagnosis.get('final_score', 0)
                trial_critique = trial_diagnosis.get('critique', '')

                group_results.append({
                    "query": variation_query,
                    "score": trial_score,
                    "critique": trial_critique,
                    "ref_img": ref_img,
                    "image_path": trial_path
                })

                f_log.write(f"    Result: Score {trial_score} | Query: {variation_query[:50]}...\n")

            if not group_results:
                f_log.write(">> Group generation failed completely. Aborting retry.\n")
                retry_cnt += 1
                continue

            # C. Analyze Group Results
            # Find best in group
            sorted_group = sorted(group_results, key=lambda x: x['score'], reverse=True)
            best_in_group = sorted_group[0]

            f_log.write(f">> Group Best: Score {best_in_group['score']} (Query: {best_in_group['query']})\n")

            # Update current image to the best of this group
            current_image = best_in_group['image_path']

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
            f_log.write(f"\n--- Final Check ---\n")
            # Check if the last current_image is better
            if os.path.exists(current_image):
                diagnosis = taxonomy_aware_diagnosis(prompt, [current_image], client, args.llm_model)
                score = diagnosis.get('score', 0)
                if score > best_score:
                    best_score = score
                    best_image_path = current_image

            f_log.write(f">> Loop finished. Best Score: {best_score}. Saving best image to FINAL.\n")
            if best_image_path and os.path.exists(best_image_path):
                shutil.copy(best_image_path, final_success_path)

        f_log.close()

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Stop Monitor & Save Plots
    monitor.stop()
    monitor.save_plots(logs_dir)

    with open(os.path.join(logs_dir, "time_elapsed.txt"), "w") as f:
        f.write(f"Total execution time: {elapsed_time:.2f} seconds\n")
        f.write(f"Total Input Tokens: {RUN_STATS['input_tokens']}\n")
        f.write(f"Total Output Tokens: {RUN_STATS['output_tokens']}\n")
        f.write(f"Total Tokens: {RUN_STATS['input_tokens'] + RUN_STATS['output_tokens']}\n")
