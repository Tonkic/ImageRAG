'''
OmniGenV2_BC_MGR_ImageNet.py
=============================
Configuration:
  - Generator: OmniGen V2
  - Critic: Binary Critic (BC)
  - Retrieval: Memory-Guided Retrieval (MGR)
  - Dataset: ImageNet (ILSVRC2012)

Usage:
  python OmniGenV2_BC_MGR_ImageNet.py \
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
import json
import shutil
import numpy as np
import torch
print(f"DEBUG: Torch sees {torch.cuda.device_count()} devices. Current device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})")
import random
from PIL import Image
from tqdm import tqdm

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="OmniGenV2 + BC + MGR (ImageNet)")

parser.add_argument("--device_id", type=int, required=True)
parser.add_argument("--task_index", type=int, default=0)
parser.add_argument("--total_chunks", type=int, default=1)
parser.add_argument("--omnigen2_path", type=str, default="./OmniGen2")
parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2")
parser.add_argument("--transformer_lora_path", type=str, default=None)
parser.add_argument("--openai_api_key", type=str, required=True)
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_retries", type=int, default=1)
parser.add_argument("--text_guidance_scale", type=float, default=7.5)
parser.add_argument("--image_guidance_scale", type=float, default=2.0)
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/imagenet")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
print(f"DEBUG: CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")
import openai
import clip

# ------------------------------------------------------------------
from binary_critic import retrieval_caption_generation  # Critic
from memory_guided_retrieval import retrieve_img_per_caption # Retrieval
from global_memory import GlobalMemory
# ------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DATASET_CONFIG = {
    "classes_txt": "datasets/imagenet_classes.txt",
    "train_list": "datasets/imagenet_train_list.txt",
    "image_root": "datasets/ILSVRC2012_train",
    "output_path": "results/OmniGenV2_BC_MGR_ImageNet"
}

def setup_system():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(script_dir, args.omnigen2_path)))

    try:
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        pipe = OmniGen2Pipeline.from_pretrained(
            args.omnigen2_model_path,
            torch_dtype=torch.bfloat16,
            transformer_lora_path=args.transformer_lora_path,
            trust_remote_code=True
        )
        if not hasattr(pipe.transformer, "enable_teacache"):
            pipe.transformer.enable_teacache = False
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.to("cuda")
    except ImportError:
        print("Error: OmniGen2 not found.")
        sys.exit(1)
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("all_proxy", None)
    client = openai.OpenAI(
        api_key=args.openai_api_key,
        base_url="https://api.siliconflow.cn/v1/"
    )
    return pipe, client

def load_retrieval_db():
    print(f"Loading ImageNet Retrieval DB...")
    paths = []
    with open(DATASET_CONFIG['train_list'], 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line: continue
            img_path = os.path.join(DATASET_CONFIG['image_root'], line)
            if os.path.exists(img_path):
                paths.append(img_path)
    print(f"Loaded {len(paths)} images.")
    return paths

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
        height=1024, width=1024,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        num_inference_steps=50,
        generator=generator
    ).images[0].save(output_path)

if __name__ == "__main__":
    seed_everything(args.seed)
    pipe, client = setup_system()
    retrieval_db = load_retrieval_db()
    os.makedirs(DATASET_CONFIG['output_path'], exist_ok=True)

    with open(DATASET_CONFIG['classes_txt'], 'r') as f:
        all_classes = [line.strip() for line in f.readlines() if line.strip()]

    my_classes = [c for i, c in enumerate(all_classes) if i % args.total_chunks == args.task_index]
    print(f"Processing {len(my_classes)} classes.")

    for class_line in tqdm(my_classes):
        parts = class_line.split(':', 1)
        if len(parts) < 2: continue
        class_id = parts[0].strip()
        class_names = parts[1].strip()
        simple_name = class_names.split(',')[0].strip()

        safe_name = f"{class_id}_{simple_name.replace(' ', '_').replace('/', '-')}"
        prompt = f"a photo of a {simple_name}"

        final_success_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_FINAL.png")
        if os.path.exists(final_success_path):
            print(f"Skipping {safe_name}: Already finished successfully.")
            continue

        log_file = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}.log")
        f_log = open(log_file, "w")
        f_log.write(f"Prompt: {prompt}\n")

        # --- Phase 1: Initial Generation ---
        v1_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V1.png")

        # [Optimization] Shared Baseline Logic
        baseline_dir = "results/OmniGenV2_Baseline_ImageNet"
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
        global_memory = GlobalMemory()
        last_used_ref = None


        while retry_cnt < args.max_retries:
            f_log.write(f"\n--- Retry {retry_cnt+1} ---\n")

            # 1. Critic (Binary)
            diagnosis = retrieval_caption_generation(
                prompt, [current_image],
                gpt_client=client, model=args.llm_model
            )

            status = diagnosis.get('status')
            f_log.write(f"Decision: {status}\n")

            # [MGR Feedback Loop]
            if last_used_ref is not None:
                is_match = (status == 'success')
                global_memory.add_feedback(last_used_ref, prompt, is_match=is_match)
                print(f"  [MGR] Feedback recorded for {os.path.basename(last_used_ref)}: {'Match' if is_match else 'Mismatch'}")

            if status == 'success':
                f_log.write(">> Success!\n")
                shutil.copy(current_image, final_success_path)
                break

            # 2. Retrieval (MGR)
            retrieved_lists, _ = retrieve_img_per_caption(
                [prompt], retrieval_db,
                embeddings_path=args.embeddings_path,
                k=50, device="cpu",
                global_memory=global_memory
            )
            candidates = retrieved_lists[0]

            if not candidates:
                f_log.write(">> No new references found in candidates.\n")
                break

            best_ref = candidates[0]
            global_memory.add(best_ref)
            last_used_ref = best_ref
            f_log.write(f">> Ref: {best_ref}\n")

            # 3. Generation
            next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")
            regen_prompt = f"{prompt}. Use this image as a reference: <|image_1|>."

            run_omnigen(pipe, regen_prompt, [best_ref], next_path, args.seed + retry_cnt + 1)

            current_image = next_path
            retry_cnt += 1

        f_log.close()


    # --- End of Class Loop ---
    print("\n============================================")
    print("All classes processed. Starting Global Memory Training...")
    try:
        # Re-initialize to ensure clean state and load all accumulated memory
        trainer_memory = GlobalMemory()
        trainer_memory.train_model(epochs=20, plot_path=os.path.join(DATASET_CONFIG['output_path'], "memory_loss.png"))
    except Exception as e:
        print(f"Error during training: {e}")
    print("============================================")