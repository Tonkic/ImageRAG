'''
OmniGenV2_BC_SR_Aircraft.py
=============================
Configuration:
  - Generator: OmniGen V2
  - Critic: Binary Critic (BC) -> Simple Success/Fail
  - Retrieval: Static Retrieval (SR) -> Top-1, No Memory
  - Dataset: FGVC-Aircraft

Usage:
  python OmniGenV2_BC_SR_Aircraft.py \
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
parser = argparse.ArgumentParser(description="OmniGenV2 + BC + SR (Aircraft)")

parser.add_argument("--device_id", type=str, required=True, help="Main device ID (e.g. '0' or '0,1')")
parser.add_argument("--vlm_device_id", type=str, default=None, help="Device ID for VLM (if different)")
parser.add_argument("--task_index", type=int, default=0)
parser.add_argument("--total_chunks", type=int, default=1)
parser.add_argument("--omnigen2_path", type=str, default="./OmniGen2")
parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2")
parser.add_argument("--transformer_lora_path", type=str, default=None)
parser.add_argument("--openai_api_key", type=str, required=False, help="Required for SiliconFlow API. If not provided, uses local model weights.")
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")

# Local Weights Config
parser.add_argument("--local_model_weight_path", type=str, default="/home/tingyu/imageRAG/Qwen3-VL-4B-Instruct")
parser.add_argument("--enable_offload", action="store_true", help="Enable CPU offloading for OmniGen to save VRAM")
parser.add_argument("--enable_teacache", action="store_true", help="Enable TeaCache acceleration")
parser.add_argument("--teacache_thresh", type=float, default=0.4, help="TeaCache relative L1 threshold")
parser.add_argument("--enable_taylorseer", action="store_true", help="Enable TaylorSeer acceleration")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--text_guidance_scale", type=float, default=7.5)
parser.add_argument("--image_guidance_scale", type=float, default=1.5)
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/aircraft")
parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=["CLIP", "LongCLIP", "SigLIP", "SigLIP2", "ColPali", "Qwen2.5-VL", "Qwen3-VL"], help="Retrieval Model")
parser.add_argument("--retrieval_datasets", nargs='+', default=['aircraft'], choices=['aircraft', 'cub', 'imagenet'], help="Datasets to use for retrieval")

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
print(f"DEBUG: Torch sees {torch.cuda.device_count()} devices. Current device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})")
import random
from PIL import Image
from tqdm import tqdm
import openai
import clip

# ------------------------------------------------------------------
# [IMPORTS] BC + SR
from binary_critic import retrieval_caption_generation  # Binary Critic
from memory_guided_retrieval import retrieve_img_per_caption   # Static Retrieval
from rag_utils import LocalQwen3VLWrapper, UsageTrackingClient, ResourceMonitor, RUN_STATS
# ------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    "output_path": f"results/{_rm}/{timestamp}/OmniGenV2_BC_SR_Aircraft_{run_time}"
}

def setup_system(omnigen_device, vlm_device_map):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(script_dir, args.omnigen2_path)))

    try:
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        pipe = OmniGen2Pipeline.from_pretrained(
            args.omnigen2_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        if not hasattr(pipe.transformer, "enable_teacache"):
            pipe.transformer.enable_teacache = False

        # Handle mutual exclusion between TeaCache and TaylorSeer
        if args.enable_teacache and args.enable_taylorseer:
            print("WARNING: enable_teacache and enable_taylorseer are mutually exclusive. TeaCache will be ignored in favor of TaylorSeer.")
            args.enable_teacache = False

        # Apply Accelerations
        if args.enable_taylorseer:
            print("Enabling TaylorSeer acceleration...")
            pipe.enable_taylorseer = True
            # Disable TeaCache explicitly just in case
            if hasattr(pipe.transformer, "enable_teacache"):
                pipe.transformer.enable_teacache = False
        elif args.enable_teacache:
            if hasattr(pipe.transformer, "enable_teacache"):
                print(f"Enabling TeaCache for OmniGen2 (threshold={args.teacache_thresh})...")
                pipe.transformer.enable_teacache = True
                pipe.transformer.rel_l1_thresh = args.teacache_thresh
            else:
                print("Warning: pipe.transformer does not have enable_teacache attribute. Ignoring.")
                pipe.transformer.enable_teacache = False
        else:
            # Disable both
            if hasattr(pipe.transformer, "enable_teacache"):
                pipe.transformer.enable_teacache = False
            pipe.enable_taylorseer = False

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

def load_retrieval_db():
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

def run_omnigen(pipe, prompt, input_images, output_path, seed):
    if isinstance(input_images, str):
        input_images = [input_images]

    processed_imgs = []
    for img in input_images:
        try:
            if isinstance(img, str): img = Image.open(img)
            if img.mode != 'RGB': img = img.convert('RGB')
            processed_imgs.append(img)
        except Exception as e:
            print(f"Error loading image: {e}")
            continue

    generator = torch.Generator(device="cuda").manual_seed(seed)

    pipe(
        prompt=prompt,
        input_images=processed_imgs,
        height=512, width=512,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        num_inference_steps=50,
        generator=generator
    ).images[0].save(output_path)

if __name__ == "__main__":
    import time
    start_time = time.time()

    seed_everything(args.seed)
    pipe, client = setup_system(omnigen_device, vlm_device_map)
    retrieval_db = load_retrieval_db()
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

    my_classes = [c for i, c in enumerate(all_classes) if i % args.total_chunks == args.task_index]
    print(f"Processing {len(my_classes)} classes.")

    for class_name in tqdm(my_classes):
        safe_name = class_name.replace(" ", "_").replace("/", "-")
        prompt = f"a photo of a {class_name}"

        log_file = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}.log")
        f_log = open(log_file, "w")
        f_log.write(f"Prompt: {prompt}\n")

        final_success_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_FINAL.png")

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

        # Loop (Static Retrieval = No Exclusion List)
        while retry_cnt < args.max_retries:
            f_log.write(f"\n--- Retry {retry_cnt+1} ---\n")

            # 1. Critic (Binary)
            diagnosis = retrieval_caption_generation(
                prompt, [current_image],
                gpt_client=client, model=args.llm_model
            )

            status = diagnosis.get('status')
            f_log.write(f"Decision: {status}\n")
            f_log.write(f"Full Diagnosis: {json.dumps(diagnosis, indent=2)}\n")

            if status == 'success':
                f_log.write(">> Success!\n")
                shutil.copy(current_image, final_success_path)
                break

            # 2. Retrieval (Static)
            # Always retrieve Top-1 based on the original prompt
            # (Since BC doesn't give us fine-grained features to augment the query)

            # [Token Length Check]
            if args.retrieval_method not in ["Qwen2.5-VL", "Qwen3-VL"]:
                from memory_guided_retrieval import check_token_length
                check_token_length([prompt], device="cpu", method=args.retrieval_method)

            retrieved_lists, _ = retrieve_img_per_caption(
                [prompt], retrieval_db,
                embeddings_path=args.embeddings_path,
                k=1, device="cuda", method=args.retrieval_method
            )

            if not retrieved_lists[0]:
                f_log.write(">> Retrieval failed.\n")
                break

            best_ref = retrieved_lists[0][0]
            f_log.write(f">> Static Ref: {best_ref}\n")

            # 3. Generation
            next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")
            regen_prompt = f"{prompt}. Use reference image <|image_1|>."

            run_omnigen(pipe, regen_prompt, [best_ref], next_path, args.seed + retry_cnt + 1)

            current_image = next_path
            retry_cnt += 1

        if not os.path.exists(final_success_path):
            f_log.write(f"\n--- Final Check (Last Generated) ---\n")
            f_log.write(f">> Loop finished. Saving last image to FINAL.\n")
            if os.path.exists(current_image):
                shutil.copy(current_image, final_success_path)

        f_log.close()


    # Stop Monitor and Save Plots
    monitor.stop()
    monitor.save_plots(logs_dir)
    print(f"Resource usage plots saved to {os.path.join(logs_dir, 'resource_usage.png')}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    with open(os.path.join(logs_dir, "time_elapsed.txt"), "w") as f:
        f.write(f"Total execution time: {elapsed_time:.2f} seconds\n")
        f.write(f"Total Input Tokens: {RUN_STATS['input_tokens']}\n")
        f.write(f"Total Output Tokens: {RUN_STATS['output_tokens']}\n")
        f.write(f"Total Tokens: {RUN_STATS['input_tokens'] + RUN_STATS['output_tokens']}\n")