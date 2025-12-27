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

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="OmniGenV2 + TAC + MGR (Aircraft)")

# Core Config
parser.add_argument("--device_id", type=int, required=True, help="GPU Device ID")
parser.add_argument("--retrieval_device_id", type=int, default=None, help="Retrieval GPU Device ID")
parser.add_argument("--task_index", type=int, default=0)
parser.add_argument("--total_chunks", type=int, default=1)

# Paths
parser.add_argument("--omnigen2_path", type=str, default="./OmniGen2")
parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2")
parser.add_argument("--transformer_lora_path", type=str, default=None)
parser.add_argument("--openai_api_key", type=str, required=True)
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")

# Generation Params
parser.add_argument("--seed", type=int, default=0, help="Global Random Seed")
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--text_guidance_scale", type=float, default=7.5)
parser.add_argument("--image_guidance_scale", type=float, default=1.5) # Higher for TAC logic
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/aircraft")
parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=["CLIP", "LongCLIP", "SigLIP", "SigLIP2", "ColPali", "Hybrid", "colqwen3"], help="Retrieval Model")

args = parser.parse_args()

# Environment
if args.retrieval_device_id is not None and args.retrieval_device_id != args.device_id:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device_id},{args.retrieval_device_id}"
    omnigen_device = "cuda:0"
    retrieval_device = "cuda:1"
    print(f"DEBUG: Using Multi-GPU. OmniGen on GPU {args.device_id} (internal cuda:0), Retrieval on GPU {args.retrieval_device_id} (internal cuda:1)")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    omnigen_device = "cuda:0"
    retrieval_device = "cuda:0"
    print(f"DEBUG: Using Single-GPU on device {args.device_id}")

print(f"DEBUG: CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")

import json
import shutil
import numpy as np
import torch
print(f"DEBUG: Torch sees {torch.cuda.device_count()} devices. Current device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})")
from PIL import Image
from tqdm import tqdm
import random
import openai
import clip

# [IMPORTS] Custom Modules
from taxonomy_aware_critic import taxonomy_aware_diagnosis # TAC Logic
from memory_guided_retrieval import retrieve_img_per_caption
from global_memory import GlobalMemory # MGR Logic

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
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        # Patch for AttributeError
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

def run_omnigen(pipe, prompt, input_images, output_path, seed, img_guidance_scale=None, text_guidance_scale=None):
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

    if text_guidance_scale is None:
        text_guidance_scale = args.text_guidance_scale

    pipe(
        prompt=prompt,
        input_images=processed_imgs,
        height=512, width=512,
        text_guidance_scale=text_guidance_scale,
        image_guidance_scale=img_guidance_scale,
        num_inference_steps=50,
        generator=generator
    ).images[0].save(output_path)

# --- 5. Main Loop ---
if __name__ == "__main__":
    import time
    start_time = time.time()

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
            device=retrieval_device,
            method=args.retrieval_method
        )
        # Clear GPU cache after retrieval model is done
        torch.cuda.empty_cache()
        print("Retrieval embeddings cached successfully.")
    except Exception as e:
        print(f"Warning during embedding caching: {e}")

    # 3. Init OmniGen (Now safe to load large model)
    pipe, client = setup_system()
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


    # 4. Load Tasks
    with open(DATASET_CONFIG['classes_txt'], 'r') as f:
        all_classes = [line.strip() for line in f.readlines() if line.strip()]

    my_classes = [c for i, c in enumerate(all_classes) if i % args.total_chunks == args.task_index]
    print(f"Processing {len(my_classes)} classes.")

    # [Global Training Data Collector]
    all_feedback_memory = []

    for class_name in tqdm(my_classes):
        safe_name = class_name.replace(" ", "_").replace("/", "-")
        prompt = f"a photo of a {class_name}"

        log_file = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}.log")
        final_success_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_FINAL.png")
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
        current_prompt = prompt
        retry_cnt = 0

        # [MGR Core]: Global Memory for Re-ranking
        global_memory = GlobalMemory()
        last_used_ref = None

        # [Score Tracking]
        best_score = -1
        best_image_path = None

        # [Knowledge Retrieval] - Sanity Check
        # Retrieve specs once per class to guide the critic
        from taxonomy_aware_critic import generate_knowledge_specs
        try:
            reference_specs = generate_knowledge_specs(class_name, client, args.llm_model)
            f_log.write(f"Reference Specs: {reference_specs}\n")
        except Exception as e:
            f_log.write(f"Reference Specs Retrieval Failed: {e}\n")
            reference_specs = None

        while retry_cnt < args.max_retries:
            f_log.write(f"\n--- Retry {retry_cnt+1} ---\n")

            # A. TAC Diagnosis
            diagnosis = taxonomy_aware_diagnosis(current_prompt, [current_image], client, args.llm_model, reference_specs=reference_specs)

            score = diagnosis.get('final_score', 0)
            taxonomy_status = diagnosis.get('taxonomy_check', 'unknown')
            critique = diagnosis.get('critique', '')
            refined_prompt = diagnosis.get('refined_prompt', current_prompt)
            mgr_queries = diagnosis.get('retrieval_queries', [class_name])

            f_log.write(f"Decision: Score {score} | Taxonomy: {taxonomy_status}\nCritique: {critique}\n")
            f_log.write(f"Full Diagnosis: {json.dumps(diagnosis, indent=2)}\n")

            # [MGR Feedback Loop]
            if last_used_ref is not None:
                # If score >= 6.0, we consider the reference "helpful/correct concept"
                is_match = (score >= 6.0)
                # global_memory.add_feedback(last_used_ref, current_prompt, is_match=is_match)
                # f_log.write(f"  [MGR] Feedback recorded for {os.path.basename(last_used_ref)}: {'Match' if is_match else 'Mismatch'} (Score: {score})\n")

            # Update Best
            if score > best_score:
                best_score = score
                best_image_path = current_image

            if score >= 8.0 or (score >= 6.0 and taxonomy_status == 'correct'):
                f_log.write(f">> Success! (Score: {score}, Taxonomy: {taxonomy_status})\n")
                shutil.copy(current_image, final_success_path)
                break

            # B. Memory-Guided Retrieval
            # Use the specific queries from TAC

            if args.retrieval_method in ["CLIP", "SigLIP", "SigLIP2"]:
                 # Use concise prompt from LLM to avoid token overflow
                 query_text = diagnosis.get('concise_retrieval_prompt', f"{class_name} {class_name}")
                 # Fallback if LLM didn't return it
                 if not query_text or len(query_text) < 5:
                     query_text = f"{class_name} {class_name}"
            else:
                 # Force Class Name injection for long-context models
                 query_text = f"{class_name} {class_name}. " + " ".join(mgr_queries)
                 if len(query_text) > 300: query_text = query_text[:300]

            # [Token Length Check]
            from memory_guided_retrieval import check_token_length
            check_token_length([query_text], device="cpu", method=args.retrieval_method)

            try:
                retrieved_lists, retrieved_scores = retrieve_img_per_caption(
                    [query_text], retrieval_db,
                    embeddings_path=args.embeddings_path,
                    k=50, device=retrieval_device, method=args.retrieval_method,
                    global_memory=global_memory
                )
                candidates = retrieved_lists[0]
                candidate_scores = retrieved_scores[0]
            except RuntimeError as e:
                f_log.write(f">> Retrieval Error: {e}\n")
                candidates = []

            if not candidates:
                f_log.write(">> No references found. Proceeding without reference.\n")
                # Fallback: Generate without reference
                next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")

                # Use refined prompt but no image
                gen_prompt = refined_prompt
                run_omnigen(pipe, gen_prompt, [], next_path, args.seed + retry_cnt + 1,
                           img_guidance_scale=args.image_guidance_scale,
                           text_guidance_scale=args.text_guidance_scale)

                current_image = next_path
                current_prompt = refined_prompt
                retry_cnt += 1
                continue

            best_ref = candidates[0]
            best_ref_score = candidate_scores[0]
            global_memory.add(best_ref)
            last_used_ref = best_ref
            f_log.write(f">> Ref: {best_ref} (Score: {best_ref_score:.4f})\n")

            # C. Generation Strategy
            next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")

            # [Modified] Unified Guidance Scale (Same as BC)
            current_img_guidance = args.image_guidance_scale # 1.5
            current_text_guidance = args.text_guidance_scale # 7.5
            f_log.write(f">> Strategy: Unified Guidance (Image: {current_img_guidance}, Text: {current_text_guidance})\n")

            # Always use refined prompt + reference
            # User Request: Use original_prompt + visual_keywords (refined_prompt)
            gen_prompt = f"{refined_prompt}. Use reference image <|image_1|>."

            run_omnigen(pipe, gen_prompt, [best_ref], next_path, args.seed + retry_cnt + 1, img_guidance_scale=current_img_guidance, text_guidance_scale=current_text_guidance)

            current_image = next_path
            # Update prompt for next round (Keep the full context)
            current_prompt = refined_prompt
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

        # Collect memory from this session
        all_feedback_memory.extend(global_memory.memory)

    # --- End of Class Loop ---
    print("\n============================================")
    print("All classes processed. Starting Global Memory Training...")
    try:
        # Re-initialize to ensure clean state and load all accumulated memory
        trainer_memory = GlobalMemory()
        trainer_memory.memory = all_feedback_memory # Inject collected memory
        trainer_memory.train_model(epochs=20, plot_path=os.path.join(DATASET_CONFIG['output_path'], "logs", "memory_loss.png"))
    except Exception as e:
        print(f"Error during training: {e}")
    print("============================================")

    end_time = time.time()
    elapsed_time = end_time - start_time
    with open(os.path.join(logs_dir, "time_elapsed.txt"), "w") as f:
        f.write(f"Total execution time: {elapsed_time:.2f} seconds\n")