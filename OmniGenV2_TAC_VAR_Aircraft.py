'''
OmniGenV2_TAC_VAR_Aircraft.py
=============================
Configuration:
  - Generator: OmniGen V2
  - Critic: Taxonomy-Aware Critic (TAC) -> Fine-grained diagnosis
  - Retrieval: VLM-As-Reranker (VAR)
    1. Base Retrieval (e.g. LongCLIP) to get Top-K (10)
    2. Qwen3-VL verifies candidates ("Does this image match...?")
    3. Use first passing candidate.
  - Dataset: FGVC-Aircraft

Usage:
  python OmniGenV2_TAC_VAR_Aircraft.py \
      --device_id 0 \
      --task_index 0 \
      --total_chunks 1 \
      --omnigen2_path ./OmniGen2 \
      --retrieval_method LongCLIP
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
parser = argparse.ArgumentParser(description="OmniGenV2 + TAC + VAR (Aircraft)")

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
parser.add_argument("--enable_teacache", action="store_true", help="Enable TeaCache acceleration")
parser.add_argument("--teacache_thresh", type=float, default=0.4, help="TeaCache relative L1 threshold")
parser.add_argument("--enable_taylorseer", action="store_true", help="Enable TaylorSeer acceleration")

# Params
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--text_guidance_scale", type=float, default=7.5)
parser.add_argument("--image_guidance_scale", type=float, default=3.0) # Updated to 3.0 for VAR
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/aircraft")
parser.add_argument("--retrieval_method", type=str, default="LongCLIP", choices=["CLIP", "LongCLIP", "SigLIP", "SigLIP2", "Qwen2.5-VL", "Qwen3-VL"], help="Base Retrieval Model")
parser.add_argument("--use_hybrid_retrieval", action="store_true", help="Enable Hybrid Retrieval (Vector + BM25)")
parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter (for Qwen2.5-VL)")
parser.add_argument("--retrieval_datasets", nargs='+', default=['aircraft'], choices=['aircraft', 'cub', 'imagenet'], help="Datasets to use for retrieval")
parser.add_argument("--var_k", type=int, default=10, help="Number of candidates for VAR reranking")

args = parser.parse_args()

# Handle Devices
if args.vlm_device_id:
    # If same device specified twice, reduce to one
    if args.device_id == args.vlm_device_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
        omnigen_device = "cuda:0"
        vlm_device_map = {"": "cuda:0"} # Same device
    else:
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

# [IMPORTS]
from taxonomy_aware_critic import taxonomy_aware_diagnosis
from memory_guided_retrieval import retrieve_img_per_caption, retrieve_composed
from rag_utils import LocalQwen3VLWrapper, UsageTrackingClient, ResourceMonitor, RUN_STATS

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
    "output_path": f"results/{_rm}/{timestamp}/OmniGenV2_TAC_VAR_Aircraft_{run_time}"
}

# --- 3. Setup ---
def setup_system(omnigen_device, vlm_device_map, shared_qwen_model=None, shared_qwen_processor=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(script_dir, args.omnigen2_path)))
    try:
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        pipe = OmniGen2Pipeline.from_pretrained(
            args.omnigen2_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Mutual Exclusion for Optimizations
        if args.enable_teacache and args.enable_taylorseer:
            print("WARNING: tea_cache and taylorseer are exclusive. Disabling tea_cache.")
            args.enable_teacache = False

        if args.enable_taylorseer:
            print("Enabling TaylorSeer acceleration...")
            pipe.enable_taylorseer = True
            if hasattr(pipe.transformer, "enable_teacache"):
                pipe.transformer.enable_teacache = False
        elif args.enable_teacache:
            if hasattr(pipe.transformer, "enable_teacache"):
                print(f"Enabling TeaCache for OmniGen2 (threshold={args.teacache_thresh})...")
                pipe.transformer.enable_teacache = True
                pipe.transformer.rel_l1_thresh = args.teacache_thresh
            else:
                 print("Warning: pipe.transformer does not have enable_teacache attribute.")
                 pipe.transformer.enable_teacache = False
        else:
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
        client = LocalQwen3VLWrapper(
            args.local_model_weight_path,
            device_map=vlm_device_map,
            shared_model=shared_qwen_model,
            shared_processor=shared_qwen_processor
        )
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
        # (Other datasets omitted for brevity as per SR script, can be copied if needed)

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
    import time
    start_time = time.time()

    seed_everything(args.seed)

    # 1. Load DB & Pre-calculate Embeddings (Warmup)
    retrieval_db = load_db()

    # [VAR] Always Pre-load Qwen3-VL for Reranking and Critic
    global GLOBAL_QWEN_MODEL, GLOBAL_QWEN_PROCESSOR
    GLOBAL_QWEN_MODEL = None
    GLOBAL_QWEN_PROCESSOR = None

    # Determine retrieval device based on VLM mapping
    retrieval_device = "cuda"
    if isinstance(vlm_device_map, dict) and "" in vlm_device_map:
        retrieval_device = vlm_device_map[""]
    print(f"[Main] Retrieval/VLM Device set to: {retrieval_device}")

    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        print(f"[Main] Loading shared Qwen3-VL for VAR & Critic...")
        GLOBAL_QWEN_PROCESSOR = AutoProcessor.from_pretrained(args.local_model_weight_path, trust_remote_code=True)
        GLOBAL_QWEN_MODEL = AutoModelForVision2Seq.from_pretrained(
            args.local_model_weight_path,
            torch_dtype=torch.bfloat16,
            device_map=vlm_device_map,
            trust_remote_code=True
        ).eval()
    except Exception as e:
        print(f"Error pre-loading Qwen3-VL: {e}")
        sys.exit(1) # Cannot run VAR without Qwen

    try:
        # Dummy Warmup
        retrieve_img_per_caption(
            ["warmup_query"],
            retrieval_db,
            embeddings_path=args.embeddings_path,
            k=1,
            device=retrieval_device,
            method=args.retrieval_method,
            # Note: GLOBAL model not needed for Base Retrieval (e.g. LongCLIP) usually,
            # but passed anyway just in case method IS Qwen.
            external_model=GLOBAL_QWEN_MODEL,
            external_processor=GLOBAL_QWEN_PROCESSOR
        )
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning during warmup: {e}")

    pipe, client = setup_system(omnigen_device, vlm_device_map, shared_qwen_model=GLOBAL_QWEN_MODEL, shared_qwen_processor=GLOBAL_QWEN_PROCESSOR)
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
                shutil.copy(baseline_v1_path, v1_path)
            else:
                run_omnigen(pipe, prompt, [], v1_path, args.seed)
                try:
                    os.makedirs(baseline_dir, exist_ok=True)
                    shutil.copy(v1_path, baseline_v1_path)
                except: pass

        current_image = v1_path
        retry_cnt = 0
        best_score = -1
        best_image_path = None

        # [Knowledge Retrieval]
        from taxonomy_aware_critic import generate_knowledge_specs
        try:
            reference_specs = generate_knowledge_specs(class_name, client, args.llm_model)
            f_log.write(f"Reference Specs: {reference_specs}\n")
        except: reference_specs = None

        while retry_cnt < args.max_retries:
            f_log.write(f"\n--- Retry {retry_cnt+1} ---\n")

            # A. TAC Diagnosis
            diagnosis = taxonomy_aware_diagnosis(prompt, [current_image], client, args.llm_model, reference_specs=reference_specs)
            score = diagnosis.get('final_score', 0)
            taxonomy_status = diagnosis.get('taxonomy_check', 'unknown')
            refined_prompt = diagnosis.get('refined_prompt', prompt)

            f_log.write(f"Decision: Score {score} | Taxonomy: {taxonomy_status}\n")

            if score > best_score:
                best_score = score
                best_image_path = current_image

            if score >= 8.0 or (score >= 6.0 and taxonomy_status == 'correct'):
                f_log.write(f">> Success! (Score: {score})\n")
                shutil.copy(current_image, os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_FINAL.png"))
                break

            # B. VAR Retrieval
            best_ref = None
            best_ref_score = 0.0

            # Use Refined Prompt
            query = refined_prompt
            if args.retrieval_method not in ["Qwen2.5-VL", "Qwen3-VL", "LongCLIP"] and len(query) > 300:
                query = query[:300]

            # Token Check
            if args.retrieval_method not in ["Qwen2.5-VL", "Qwen3-VL", "LongCLIP"]:
                from memory_guided_retrieval import check_token_length
                check_token_length([query], device="cpu", method=args.retrieval_method)

            try:
                # 1. Base Retrieval (Top-K)
                k_candi = args.var_k
                retrieved_lists, retrieved_scores = retrieve_img_per_caption(
                    [query], retrieval_db,
                    embeddings_path=args.embeddings_path,
                    k=k_candi, device=retrieval_device, method=args.retrieval_method,
                    use_hybrid=args.use_hybrid_retrieval,
                    external_model=GLOBAL_QWEN_MODEL,
                    external_processor=GLOBAL_QWEN_PROCESSOR
                )

                if retrieved_lists and retrieved_lists[0]:
                    candidates = retrieved_lists[0]
                    f_log.write(f">> [VAR] Base Retrieval ({args.retrieval_method}) found {len(candidates)} candidates. Reranking...\n")

                    found_match = False
                    for idx, c_path in enumerate(candidates):
                         # 2. VLM Rerank/Verification
                         check_q = f"Does this image match the description: \"{query}\"? Answer Yes or No."
                         try:
                             # Use client for shared model access
                             resp = client.chat.completions.create(
                                 model=args.llm_model,
                                 messages=[{
                                     "role": "user",
                                     "content": [
                                         {"type": "text", "text": check_q},
                                         {"type": "image_url", "image_url": {"url": c_path}}
                                     ]
                                 }],
                                 temperature=0.01
                             )
                             ans = resp.choices[0].message.content
                             if "Yes" in ans or "yes" in ans.lower():
                                 best_ref = c_path
                                 best_ref_score = retrieved_scores[0][idx]
                                 f_log.write(f"   [PASS] Candidate {idx}: {ans} ({os.path.basename(c_path)})\n")
                                 found_match = True
                                 break # Pick first passing
                             else:
                                 f_log.write(f"   [FAIL] Candidate {idx}: {ans}\n")
                         except Exception as ce:
                             f_log.write(f"   [ERR] Candidate {idx} check error: {ce}\n")

                    if not found_match:
                         f_log.write(">> [VAR] No candidate passed verification. Fallback to Top-1.\n")
                         best_ref = candidates[0]
                         best_ref_score = retrieved_scores[0][0]

                else:
                    raise ValueError("Empty retrieval result")

            except Exception as e:
                f_log.write(f">> Retrieval Error: {e}\n")
                if retrieval_db:
                    best_ref = random.choice(retrieval_db)
                    f_log.write(f">> Using Random: {best_ref}\n")

            f_log.write(f">> Selected Ref: {best_ref} (Score: {best_ref_score:.4f})\n")

            # C. Generation
            next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")

            # Use Fixed Guidance 3.0
            current_img_guidance = args.image_guidance_scale
            current_text_guidance = args.text_guidance_scale

            gen_prompt = f"{prompt}. Use reference image <|image_1|>."
            run_omnigen(pipe, gen_prompt, [best_ref], next_path, args.seed + retry_cnt + 1,
                        img_guidance_scale=current_img_guidance, text_guidance_scale=current_text_guidance)

            current_image = next_path
            retry_cnt += 1

        # Final Check
        final_success_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_FINAL.png")
        if not os.path.exists(final_success_path):
             if os.path.exists(current_image):
                diagnosis = taxonomy_aware_diagnosis(prompt, [current_image], client, args.llm_model)
                score = diagnosis.get('score', 0)
                if score > best_score:
                    best_image_path = current_image
                if best_image_path and os.path.exists(best_image_path):
                     shutil.copy(best_image_path, final_success_path)

        f_log.close()

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
