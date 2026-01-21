'''
OmniGenV2_BC_SR_CUB.py
=============================
Configuration:
  - Generator: OmniGen V2
  - Critic: Binary Critic (BC) -> Simple Success/Fail
  - Retrieval: Static Retrieval (SR) -> Top-1, No Memory
  - Dataset: CUB-200-2011 (Birds)

Usage:
  python OmniGenV2_BC_SR_CUB.py \
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
parser = argparse.ArgumentParser(description="OmniGenV2 + BC + SR (CUB)")

parser.add_argument("--device_id", type=int, required=True)
parser.add_argument("--task_index", type=int, default=0)
parser.add_argument("--total_chunks", type=int, default=1)
parser.add_argument("--omnigen2_path", type=str, default="./OmniGen2")
parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2")
parser.add_argument("--transformer_lora_path", type=str, default=None)
parser.add_argument("--openai_api_key", type=str, required=False, help="Required for SiliconFlow API. If not provided, uses local model weights.")
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")

# Local Weights Config
parser.add_argument("--use_local_model_weight", action="store_true", help="Load local model weights directly (transformers)")
parser.add_argument("--local_model_weight_path", type=str, default="/home/tingyu/imageRAG/Qwen3-VL-4B-Instruct")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--text_guidance_scale", type=float, default=7.5)
parser.add_argument("--image_guidance_scale", type=float, default=1.5)
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/cub")
parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=["CLIP", "LongCLIP", "SigLIP", "SigLIP2", "ColPali", "Qwen2.5-VL", "Qwen3-VL"], help="Retrieval Model")

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

# ------------------------------------------------------------------
# [IMPORTS] BC + SR
from binary_critic import retrieval_caption_generation  # Binary Critic
from memory_guided_retrieval import retrieve_img_per_caption   # Static Retrieval
from rag_utils import ResourceMonitor, RUN_STATS, UsageTrackingClient, LocalQwen3VLWrapper
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
    "classes_txt": "datasets/CUB_200_2011/classes.txt",
    "train_list": "datasets/CUB_200_2011/train_test_split.txt",
    "image_root": "datasets/CUB_200_2011/images",
    "output_path": f"results/{_rm}/{timestamp}/OmniGenV2_BC_SR_CUB_{run_time}"
}

def setup_system():
    # Start Resource Monitor
    monitor = ResourceMonitor(interval=1)
    monitor.start()

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
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.to("cuda")
    except ImportError as e:
        print(f"Error: OmniGen2 not found. Details: {e}")
        sys.exit(1)
Client...")
    # Logic: Missing API Key -> Use Local Weights
    if not args.openai_api_key:
        print(f"  Using Local Model Weights from {args.local_model_weight_path}")
        client = LocalQwen3VLWrapper(args.local_model_weight_path, device_map="auto")
        # Override llm_model arg to avoid confusion
        args.llm_model = "local-qwen3-vl"
    else:
        print("  Using SiliconFlow API...")
        client = openai.OpenAI(
            api_key=args.openai_api_key,
            base_url="https://api.siliconflow.cn/v1/"
            base_url="https://api.siliconflow.cn/v1/"
    )
    client = UsageTrackingClient(client)
    return pipe, client, monitor

def load_db():
    print("Loading CUB DB...")
    # 1. Load Train Split
    train_ids = set()
    with open(DATASET_CONFIG['train_list'], 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == '1':
                train_ids.add(parts[0])

    # 2. Load Image Paths
    paths = []
    # images.txt is in the parent directory of image_root (datasets/CUB_200_2011/images.txt)
    images_txt = os.path.join(os.path.dirname(DATASET_CONFIG['image_root']), "images.txt")

    with open(images_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_id = parts[0]
                rel_path = parts[1]
                if img_id in train_ids:
                    full_path = os.path.join(DATASET_CONFIG['image_root'], rel_path)
                    if os.path.exists(full_path):
                        paths.append(full_path)
    print(f"Loaded {len(paths)} training images.")
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
        height=512, width=512,
        text_guidance_scale=text_guidance_scale,
        image_guidance_scale=img_guidance_scale,
        num_inference_steps=50,
        generator=gen
    ).images[0].save(output_path)

# --- Main ---
if __name__ == "__main__":
    import time
    start_time = time.time()

    seed_everything(args.seed)

    # 1. Load DB & Pre-calculate Embeddings
    retrieval_db = load_db()
    print("Pre-calculating/Loading retrieval embeddings on GPU...")
    try:
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

    pipe, client, monitor = setup_system()
    os.makedirs(DATASET_CONFIG['output_path'], exist_ok=True)

    # Create logs directory
    logs_dir = os.path.join(DATASET_CONFIG['output_path'], "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Save Run Configuration
    config_path = os.path.join(logs_dir, "run_config.txt")
    with open(config_path, "w") as f:
        f.write("Run Configuration:\n")
        f.write("==================\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write(f"\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")

    # Load Classes
    with open(DATASET_CONFIG['classes_txt']) as f:
        all_classes = []
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # Format: 1 001.Black_footed_Albatross
                raw_name = parts[1]
                if '.' in raw_name:
                    raw_name = raw_name.split('.', 1)[1]
                all_classes.append(raw_name)

    my_tasks = [c for i, c in enumerate(all_classes) if i % args.total_chunks == args.task_index]
    print(f"Processing {len(my_tasks)} classes.")

    for class_name_raw in tqdm(my_tasks):
        class_name = class_name_raw.replace("_", " ")
        safe_name = class_name_raw # Use original with underscores for filenames
        prompt = f"a photo of a {class_name}, a type of bird"

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
                shutil.copy(baseline_v1_path, v1_path)
            else:
                run_omnigen(pipe, prompt, [], v1_path, args.seed)
                try:
                    os.makedirs(baseline_dir, exist_ok=True)
                    shutil.copy(v1_path, baseline_v1_path)
                except: pass

        current_image = v1_path
        retry_cnt = 0

        while retry_cnt < args.max_retries:
            f_log.write(f"\n--- Retry {retry_cnt+1} ---\n")

            # 1. Binary Critic
            diagnosis = retrieval_caption_generation(prompt, [current_image], client, args.llm_model)
            status = diagnosis.get('status', 'fail')
            critique = diagnosis.get('critique', '')
            refined_prompt = diagnosis.get('refined_prompt', prompt)

            f_log.write(f"Decision: {status}\nCritique: {critique}\n")
            f_log.write(f"Full Diagnosis: {json.dumps(diagnosis, indent=2)}\n")

            if status == 'success':
                f_log.write(">> Success!\n")
                shutil.copy(current_image, os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_FINAL.png"))
                break

            # 2. Static Retrieval
            # Force Class Name injection
            query = f"{class_name}. {refined_prompt}"
            if args.retrieval_method not in ["Qwen2.5-VL", "Qwen3-VL", "LongCLIP"]:
                if len(query) > 300: query = query[:300]

            # [Token Length Check]
            if args.retrieval_method not in ["Qwen2.5-VL", "Qwen3-VL", "LongCLIP"]:
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
                    k=1, device="cuda:0", method=args.retrieval_method
                )
                best_ref = retrieved_lists[0][0]
                best_ref_score = retrieved_scores[0][0]

            f_log.write(f">> Static Ref: {best_ref} (Score: {best_ref_score:.4f})\n")

            # 3. Generation
            next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")

            # Unified Guidance
            current_img_guidance = 1.5
            current_text_guidance = args.text_guidance_scale
            f_log.write(f">> Strategy: Unified Guidance (Image: {current_img_guidance}, Text: {current_text_guidance})\n")

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
                diagnosis = retrieval_caption_generation(prompt, [current_image], client, args.llm_model)
                status = diagnosis.get('status', 'fail')
                if status == 'success':
                    f_log.write(">> Success!\n")
                    shutil.copy(current_image, final_success_path)
                else:
                    f_log.write(">> Failed after retries. Saving last image as FINAL.\n")
                    shutil.copy(current_image, final_success_path)

        f_log.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    with open(os.path.join(logs_dir, "time_elapsed.txt"), "w") as f:
        f.write(f"Total execution time: {elapsed_time:.2f} seconds\n")
        f.write(f"Total Input Tokens: {RUN_STATS['input_tokens']}\n")
        f.write(f"Total Output Tokens: {RUN_STATS['output_tokens']}\n")
        f.write(f"Total Tokens: {RUN_STATS['input_tokens'] + RUN_STATS['output_tokens']}\n")
        f.write(f"Total Input Tokens: {RUN_STATS['input_tokens']}\n")
        f.write(f"Total Output Tokens: {RUN_STATS['output_tokens']}\n")
        f.write(f"Total Tokens: {RUN_STATS['input_tokens'] + RUN_STATS['output_tokens']}\n")

    # Stop Monitor
    monitor.stop()
    monitor.save_plots(os.path.join(DATASET_CONFIG['output_path'], "resource_usage.png"))
