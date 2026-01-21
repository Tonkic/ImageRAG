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
from datetime import datetime


import argparse
import sys
import os
import torchvision.transforms as transforms
# [Proxy Config] Clear system proxies for direct connection
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="OmniGenV2 + TAC + SR (Aircraft)")

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
parser.add_argument("--image_guidance_scale", type=float, default=1.5) # Higher guidance for composition
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/aircraft")
parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=["CLIP", "LongCLIP", "SigLIP", "SigLIP2", "Qwen2.5-VL", "Qwen3-VL"], help="Retrieval Model")
parser.add_argument("--use_hybrid_retrieval", action="store_true", help="Enable Hybrid Retrieval (Vector + BM25)")
parser.add_argument("--enable_pic2word", action="store_true", help="Enable Pic2Word Composed Retrieval (requires previous image)")
parser.add_argument("--pic2word_checkpoint", type=str, default="pic2word_mapper_ep9.pt", help="Path to Pic2Word Mapper checkpoint")
parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter (for Qwen2.5-VL)")
parser.add_argument("--birefnet_model_path", type=str, default="ZhengPeng7/BiRefNet", help="Path or HF ID for BiRefNet")
parser.add_argument("--retrieval_datasets", nargs='+', default=['aircraft'], choices=['aircraft', 'cub', 'imagenet'], help="Datasets to use for retrieval")

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
from taxonomy_aware_critic import taxonomy_aware_diagnosis # The new Critic
from memory_guided_retrieval import retrieve_img_per_caption, retrieve_composed  # The Static Retrieval logic + Composed
from rag_utils import LocalQwen3VLWrapper, UsageTrackingClient, ResourceMonitor, RUN_STATS

# --- BiRefNet Utilities ---
GLOBAL_BIREFNET = None
def load_birefnet(model_path, device):
    global GLOBAL_BIREFNET
    if GLOBAL_BIREFNET is None:
        print(f"Loading BiRefNet from {model_path} (Manual Load)...")
        try:
            from transformers import AutoConfig
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            from huggingface_hub import hf_hub_download

            # [Fix 5.0] Manual instantiation to bypass Accelerate's Meta Tensor initialization

            # 1. Load Config & Class
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model_class = get_class_from_dynamic_module(
                config.auto_map["AutoModelForImageSegmentation"],
                model_path
            )

            # 2. Instantiate on CPU
            model = model_class(config)

            # 3. Load Weights
            weight_path = None
            state_dict = None
            try:
                # Try safetensors
                weight_path = hf_hub_download(repo_id=model_path, filename="model.safetensors")
                from safetensors.torch import load_file
                state_dict = load_file(weight_path)
            except Exception:
                try:
                    # Fallback to pytorch_model.bin
                    weight_path = hf_hub_download(repo_id=model_path, filename="pytorch_model.bin")
                    state_dict = torch.load(weight_path, map_location="cpu")
                except Exception as e_load:
                    print(f"Could not find weights for BiRefNet: {e_load}")
                    raise e_load

            # 4. Apply Weights
            msg = model.load_state_dict(state_dict, strict=False) # strict=False to be safe with version mismatches
            print(f"BiRefNet Weights loaded: {msg}")

            # 5. Move to Device
            GLOBAL_BIREFNET = model
            GLOBAL_BIREFNET.to(device)
            GLOBAL_BIREFNET.eval()

        except Exception as e:
            print(f"Error loading BiRefNet: {e}")
            import traceback
            traceback.print_exc()
            GLOBAL_BIREFNET = None
    return GLOBAL_BIREFNET

def remove_background(image, model, device):
    # Prepare Input
    w, h = image.size
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_images = transform_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize((w, h))

    # Apply Mask
    out_img = image.copy()
    out_img.putalpha(mask)

    # Composite on White Background
    final_image = Image.new("RGB", image.size, (255, 255, 255))
    final_image.paste(out_img, (0, 0), out_img)

    return final_image

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
    "output_path": f"results/{_rm}/{timestamp}/OmniGenV2_TAC_SR_Aircraft_{run_time}"
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
            trust_remote_code=True, # Required for custom code
        )
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

    # os.environ.pop("http_proxy", None)
    # os.environ.pop("https_proxy", None)
    # os.environ.pop("all_proxy", None)

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
    import time
    start_time = time.time()

    seed_everything(args.seed)

    # 1. Load DB & Pre-calculate Embeddings (Warmup)
    retrieval_db = load_db()
    print("Pre-calculating/Loading retrieval embeddings on GPU...")

    # [Optimization] Pre-load Qwen3-VL here if retrieval method matches, then share it
    global GLOBAL_QWEN_MODEL, GLOBAL_QWEN_PROCESSOR
    GLOBAL_QWEN_MODEL = None
    GLOBAL_QWEN_PROCESSOR = None

    # Determine retrieval device based on VLM mapping
    retrieval_device = "cuda"
    if isinstance(vlm_device_map, dict) and "" in vlm_device_map:
        retrieval_device = vlm_device_map[""]
    print(f"[Main] Retrieval/VLM Device set to: {retrieval_device}")

    if args.retrieval_method == "Qwen3-VL":
         try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            print(f"[Main] Loading shared Qwen3-VL for Retrieval & Critic...")
            GLOBAL_QWEN_PROCESSOR = AutoProcessor.from_pretrained(args.local_model_weight_path, trust_remote_code=True)
            GLOBAL_QWEN_MODEL = AutoModelForVision2Seq.from_pretrained(
                args.local_model_weight_path,
                torch_dtype=torch.bfloat16,
                device_map=vlm_device_map, # Share same device map
                trust_remote_code=True
            )

            # [Adapter Loading]
            if args.adapter_path:
                try:
                    from peft import PeftModel
                    print(f"[Main] Loading LoRA adapter for Qwen3-VL from {args.adapter_path}...")
                    GLOBAL_QWEN_MODEL = PeftModel.from_pretrained(GLOBAL_QWEN_MODEL, args.adapter_path)
                except ImportError:
                    print("[Main] Warning: PEFT not installed, skipping adapter.")
                except Exception as e:
                    print(f"[Main] Error loading adapter: {e}")

            GLOBAL_QWEN_MODEL = GLOBAL_QWEN_MODEL.eval()

         except Exception as e:
            print(f"Error pre-loading Qwen3-VL: {e}")

    try:
        # Run a dummy retrieval on the FULL database to force caching
        retrieve_img_per_caption(
            ["warmup_query"],
            retrieval_db,
            embeddings_path=args.embeddings_path,
            k=1,
            device=retrieval_device, # Use correct device
            method=args.retrieval_method,
            # [Fix] Explicitly pass shared components to prevent reloading
            external_model=GLOBAL_QWEN_MODEL,
            external_processor=GLOBAL_QWEN_PROCESSOR
        )
        torch.cuda.empty_cache()
        print("Retrieval embeddings cached successfully.")
    except Exception as e:
        print(f"Warning during embedding caching: {e}")

    pipe, client = setup_system(omnigen_device, vlm_device_map, shared_qwen_model=GLOBAL_QWEN_MODEL, shared_qwen_processor=GLOBAL_QWEN_PROCESSOR)
    # retrieval_db = load_db() # Already loaded
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

        f.write("\n[Command Line Arguments]\n")
        for arg in sorted(vars(args)):
            f.write(f"{arg}: {getattr(args, arg)}\n")

        f.write("\n[Dataset Configuration]\n")
        for k, v in DATASET_CONFIG.items():
            f.write(f"{k}: {v}\n")

        f.write("\n[System & Device Configuration]\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        f.write(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}\n")
        f.write(f"OmniGen Device (Internal): {omnigen_device}\n")
        f.write(f"VLM Device Map: {vlm_device_map}\n")
        f.write(f"Retrieval Device: {retrieval_device}\n")

        try:
            import transformers
            f.write(f"Transformers Version: {transformers.__version__}\n")
            f.write(f"PyTorch Version: {torch.__version__}\n")
            if torch.cuda.is_available():
                f.write(f"GPU Name: {torch.cuda.get_device_name(0)}\n")

            # [Check Adapter Status]
            f.write("\n[Runtime Model Validation]\n")
            if GLOBAL_QWEN_MODEL:
                f.write(f"Global Qwen Model Loaded: YES\n")
                # Check if it is a PeftModel
                is_peft = False
                try:
                    from peft import PeftModel
                    if isinstance(GLOBAL_QWEN_MODEL, PeftModel):
                        is_peft = True
                        f.write(f"Adapter Active: YES\n")
                        f.write(f"Adapter Config: {GLOBAL_QWEN_MODEL.peft_config}\n")
                    else:
                        f.write(f"Adapter Active: NO (Model is raw {type(GLOBAL_QWEN_MODEL).__name__})\n")
                except ImportError:
                    f.write(f"Adapter Check Skipped (peft not imported)\n")
            else:
                f.write(f"Global Qwen Model Loaded: NO (Using on-demand loading?)\n")

        except: pass


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

        # [Knowledge Retrieval] - Sanity Check
        from taxonomy_aware_critic import generate_knowledge_specs
        try:
            reference_specs = generate_knowledge_specs(class_name, client, args.llm_model)
            f_log.write(f"Reference Specs: {reference_specs}\n")
        except Exception as e:
            f_log.write(f"Reference Specs Retrieval Failed: {e}\n")
            reference_specs = None

        # Loop (Static Retrieval = No Exclusion List)
        while retry_cnt < args.max_retries:
            f_log.write(f"\n--- Retry {retry_cnt+1} ---\n")

            # A. TAC Diagnosis (V4.0)
            diagnosis = taxonomy_aware_diagnosis(prompt, [current_image], client, args.llm_model, reference_specs=reference_specs)

            score = diagnosis.get('final_score', 0)
            taxonomy_status = diagnosis.get('taxonomy_check', 'unknown')
            critique = diagnosis.get('critique', '')
            refined_prompt = diagnosis.get('refined_prompt', prompt)

            f_log.write(f"Decision: Score {score} | Taxonomy: {taxonomy_status}\nCritique: {critique}\n")
            f_log.write(f"Full Diagnosis: {json.dumps(diagnosis, indent=2)}\n")

            # Update Best
            if score > best_score:
                best_score = score
                best_image_path = current_image

            if score >= 8.0 or (score >= 6.0 and taxonomy_status == 'correct'):
                f_log.write(f">> Success! (Score: {score}, Taxonomy: {taxonomy_status})\n")
                shutil.copy(current_image, os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_FINAL.png"))
                break

            # B. Static Retrieval (Simplified for SR)
            best_ref = None
            best_ref_score = 0.0
            search_performed = False

            # [Composed Retrieval] Uses Pic2Word if enabled and we have a reference image
            if retry_cnt > 0 and os.path.exists(current_image) and args.enable_pic2word:
                f_log.write(f">> Using Pic2Word Composed Retrieval (Method: {args.retrieval_method})...\n")
                try:
                    c_paths, c_scores = retrieve_composed(
                        ref_image_path=current_image,
                        modifier_text=refined_prompt,
                        image_paths=retrieval_db,
                        embeddings_path=args.embeddings_path,
                        k=1,
                        device=retrieval_device, # Use configured device
                        method=args.retrieval_method,
                        mapper_path=args.pic2word_checkpoint
                    )
                    if c_paths:
                        best_ref = c_paths[0]
                        best_ref_score = c_scores[0]
                        search_performed = True
                    else:
                        f_log.write(">> Composed Retrieval returned no results.\n")
                except Exception as e:
                    f_log.write(f">> Composed Retrieval error: {e}\n")

            if not search_performed:
                # [Modified] Simplify Retrieval Query - 显式包含 class_name
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
                        k=1, device=retrieval_device, method=args.retrieval_method,
                        adapter_path=args.adapter_path,
                        use_hybrid=args.use_hybrid_retrieval,
                        external_model=GLOBAL_QWEN_MODEL,
                        external_processor=GLOBAL_QWEN_PROCESSOR
                    )
                    if retrieved_lists and retrieved_lists[0]:
                        best_ref = retrieved_lists[0][0]
                        best_ref_score = retrieved_scores[0][0]
                    else:
                        raise ValueError("Empty retrieval result")
                except Exception as e:
                    f_log.write(f">> Retrieval Error (Query): {e}\n")
                    # Fallback to prompt
                    try:
                        retrieved_lists, retrieved_scores = retrieve_img_per_caption(
                            [prompt], retrieval_db,
                            embeddings_path=args.embeddings_path,
                            k=1, device=retrieval_device, method=args.retrieval_method,
                            adapter_path=args.adapter_path,
                            use_hybrid=args.use_hybrid_retrieval,
                            external_model=GLOBAL_QWEN_MODEL,
                            external_processor=GLOBAL_QWEN_PROCESSOR
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

            f_log.write(f">> Static Ref: {best_ref} (Score: {best_ref_score:.4f})\n")

            # [BiRefNet Background Removal]
            ref_input = best_ref
            nobg_cache_dir = "datasets/nobg_cache"
            os.makedirs(nobg_cache_dir, exist_ok=True)

            if best_ref:
                try:
                    # 1. Determine Cache Path
                    cache_path = None
                    if isinstance(best_ref, str):
                        # Use filename as unique key (e.g., 1047583.png)
                        clean_name = os.path.basename(best_ref).rsplit('.', 1)[0]
                        cache_path = os.path.join(nobg_cache_dir, f"{clean_name}.png")

                    # 2. Check Cache
                    if cache_path and os.path.exists(cache_path):
                        f_log.write(f">> BiRefNet: Using Cached No-BG Image: {cache_path}\n")
                        print(f"BiRefNet: Hit Cache -> {cache_path}")
                        ref_input = Image.open(cache_path) # Load from cache
                    else:
                        # 3. Process with BiRefNet (Only if not cached)
                        # Load BiRefNet (Lazy Load on first use)
                        bg_device = retrieval_device if retrieval_device != "cpu" else "cuda"
                        birefnet = load_birefnet(args.birefnet_model_path, bg_device)

                        if birefnet:
                            if isinstance(best_ref, str):
                                ref_img_pil = Image.open(best_ref).convert("RGB")
                            else:
                                # It's already a PIL Image
                                ref_img_pil = best_ref.convert("RGB")

                            f_log.write(f">> Removing background using BiRefNet...\n")
                            ref_img_nobg = remove_background(ref_img_pil, birefnet, bg_device)

                            # 4. Save to Cache
                            if cache_path:
                                try:
                                    ref_img_nobg.save(cache_path)
                                    f_log.write(f">> BiRefNet: Cached result to {cache_path}\n")
                                except Exception as e_save:
                                    f_log.write(f">> BiRefNet Cache Save Error: {e_save}\n")

                            ref_input = ref_img_nobg

                        # Optional: Save nobg image for debug (current run context)
                        nobg_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}_ref_nobg.png")
                        if isinstance(ref_input, Image.Image):
                            ref_input.save(nobg_path)
                except Exception as e_bg:
                    f_log.write(f">> BiRefNet Error: {e_bg}\n")

            # C. Generation Strategy
            next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")

            # [Modified] Unified Guidance Scale (Same as BC)
            current_img_guidance = 1.5 # Fixed value as requested
            current_text_guidance = args.text_guidance_scale
            f_log.write(f">> Strategy: Unified Guidance (Image: {current_img_guidance}, Text: {current_text_guidance})\n")

            # [Modified] Simplify Generation Prompt: Only use core prompt + reference trigger
            if refined_prompt != prompt:
                # 如果有 TAC 的改进意见，同时包含原始 prompt 和改进意见
                gen_prompt = f"{prompt}. {refined_prompt}. Use reference image <|image_1|>."
                f_log.write(f">> Gen Prompt (Reinforced): {gen_prompt}\n")
            else:
                gen_prompt = f"{prompt}. Use reference image <|image_1|>."

            run_omnigen(pipe, gen_prompt, [ref_input], next_path, args.seed + retry_cnt + 1, img_guidance_scale=current_img_guidance, text_guidance_scale=current_text_guidance)

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