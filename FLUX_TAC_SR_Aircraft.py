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
# Helps with fragmentation on tight GPU memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="FLUX + TAC + SR (Aircraft)")

# Core Config
parser.add_argument("--device_id", type=str, required=True, help="Main device ID (e.g. '0' or '0,1')")
parser.add_argument("--vlm_device_id", type=str, default=None, help="Device ID for VLM (if different)")
parser.add_argument("--retrieval_device_id", type=str, default=None, help="Device ID for Retrieval (if different)")
parser.add_argument("--task_index", type=int, default=0)
parser.add_argument("--total_chunks", type=int, default=1)

# Paths
parser.add_argument("--openai_api_key", type=str, required=False, help="Required for SiliconFlow API. If not provided, uses local model weights.")
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")

# Local Weights Config
parser.add_argument("--use_local_model_weight", action="store_true", help="Load local model weights directly (transformers)")
parser.add_argument("--local_model_weight_path", type=str, default="/home/tingyu/imageRAG/Qwen3-VL-4B-Instruct")
parser.add_argument("--enable_offload", action="store_true", help="Enable CPU offloading to save VRAM (default: False for speed)")

# Params
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--text_guidance_scale", type=float, default=30.0) # FLUX default
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/aircraft")
parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=["CLIP", "LongCLIP", "SigLIP", "SigLIP2", "ColPali", "Qwen2.5-VL", "Qwen3-VL"], help="Retrieval Model")
parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter (for Qwen2.5-VL)")
parser.add_argument("--retrieval_datasets", nargs='+', default=['aircraft'], choices=['aircraft', 'cub', 'imagenet'], help="Datasets to use for retrieval")

args = parser.parse_args()

# Handle Devices
# Collect all requested device IDs (preserving order and uniqueness)
device_list = [args.device_id]
if args.vlm_device_id and args.vlm_device_id not in device_list:
    device_list.append(args.vlm_device_id)
if args.retrieval_device_id and args.retrieval_device_id not in device_list:
    device_list.append(args.retrieval_device_id)

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_list)
print(f"DEBUG: CUDA_VISIBLE_DEVICES set to {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Helper to find logical index
def get_logical_device(phys_id):
    try:
        idx = device_list.index(phys_id)
        return f"cuda:{idx}"
    except ValueError:
        return "cuda:0"

flux_device = get_logical_device(args.device_id) # Should be cuda:0 usually

if args.vlm_device_id:
    vlm_device_map = {"": get_logical_device(args.vlm_device_id)}
elif args.retrieval_device_id and args.retrieval_device_id != args.device_id:
    # If a separate retrieval device is defined, put VLM there to save memory on Generation GPU
    vlm_target = get_logical_device(args.retrieval_device_id)
    print(f"DEBUG: Auto-assigning VLM to Retrieval Device ({vlm_target}) to save generation VRAM.")
    vlm_device_map = {"": vlm_target}
else:
    vlm_device_map = "auto"

if args.retrieval_device_id:
    retrieval_device = get_logical_device(args.retrieval_device_id)
else:
    # retrieval_device = "cuda:0" # Default retrieval to first available device (likely shared with Flux)
    # If using separate devices and no specific retrieval ID, try to not collide with Flux
    if len(device_list) > 1 and flux_device == "cuda:0":
         retrieval_device = "cuda:1"
    else:
         retrieval_device = "cuda:0"

print(f"DEBUG: Flux Device: {flux_device}, VLM Map: {vlm_device_map}, Retrieval Device: {retrieval_device}")

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
from memory_guided_retrieval import retrieve_img_per_caption      # The Static Retrieval logic imported from memory module like Reference
from rag_utils import LocalQwen3VLWrapper, UsageTrackingClient, ResourceMonitor, RUN_STATS
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from PIL import Image, ImageDraw

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
    "output_path": f"results/{_rm}/{timestamp}/FLUX_TAC_SR_Aircraft_{run_time}"
}

# --- 3. Setup ---
def setup_system(flux_device, vlm_device_map):
    print("Loading FLUX.1-Fill-dev...")
    pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)

    if args.enable_offload:
        print("Enabling model CPU offload for FLUX (Optimized for 24GB VRAM)...")
        pipe.enable_model_cpu_offload(device=flux_device)
    else:
        print(f"Moving FLUX to {flux_device} (Full Load)...")
        try:
            pipe.to(flux_device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n" + "="*60)
                print(" [WARNING] GPU VRAM Insufficient for Full Load (bfloat16 needs ~23.8GB).")
                print(f" [WARNING] Falling back to 'enable_model_cpu_offload({flux_device})'.")
                print(" [WARNING] This keeps the model usable on 24GB cards automatically.")
                print("="*60 + "\n")
                # Clear partial allocation
                torch.cuda.empty_cache()
                pipe.enable_model_cpu_offload(device=flux_device)
            else:
                raise e

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
    import time
    start_time = time.time()

    seed_everything(args.seed)

    # 1. Load DB
    retrieval_db = load_db()

    # 2. Setup System (Load FLUX first to prioritize VRAM)
    pipe, client = setup_system(flux_device, vlm_device_map)

    # 3. Pre-calculate Embeddings (Warmup)
    # Using the designated retrieval device
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
                    k=1, device=retrieval_device, method=args.retrieval_method
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
                        k=1, device=retrieval_device, method=args.retrieval_method
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
