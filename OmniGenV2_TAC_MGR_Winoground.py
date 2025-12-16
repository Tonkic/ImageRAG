'''
OmniGenV2_TAC_MGR_Winoground.py
=============================
Configuration:
  - Generator: OmniGen V2
  - Critic: Taxonomy Aware Critic (TAC)
  - Retrieval: Memory-Guided Retrieval (MGR)
  - Dataset: Winoground
'''

import argparse
import sys
import os


# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="OmniGenV2 + TAC + MGR (Winoground)")

parser.add_argument("--device_id", type=int, required=True)
parser.add_argument("--task_index", type=int, default=0)
parser.add_argument("--total_chunks", type=int, default=1)
parser.add_argument("--omnigen2_path", type=str, default="./OmniGen2")
parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2")
parser.add_argument("--transformer_lora_path", type=str, default=None)
parser.add_argument("--openai_api_key", type=str, required=True)
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--text_guidance_scale", type=float, default=7.5)
parser.add_argument("--image_guidance_scale", type=float, default=1.5)
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/winoground")
parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=["CLIP", "LongCLIP", "SigLIP", "ColPali", "Hybrid"], help="Retrieval Model")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
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
from datasets import load_dataset
import time

# ------------------------------------------------------------------
from taxonomy_aware_critic import taxonomy_aware_diagnosis  # Critic
from memory_guided_retrieval import retrieve_img_per_caption, check_token_length
from global_memory import GlobalMemory # Retrieval
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
    "output_path": "results/OmniGenV2_TAC_MGR_Winoground",
    "image_root": "datasets/winoground/images"
}

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
        if not hasattr(pipe.transformer, "enable_teacache"):
            pipe.transformer.enable_teacache = False
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        if args.transformer_lora_path:
            print(f"Loading LoRA from {args.transformer_lora_path}")
            pipe.load_lora_weights(args.transformer_lora_path, adapter_name="lora")
            pipe.set_adapters(["lora"])

        pipe.to("cuda")
    except Exception as e:
        print(f"Error loading OmniGenV2: {e}")
        sys.exit(1)

    client = openai.OpenAI(
        api_key=args.openai_api_key,
        base_url="https://api.siliconflow.cn/v1/"
    )
    return pipe, client

def load_winoground_db():
    print("Loading Winoground DB...")
    os.makedirs(DATASET_CONFIG['image_root'], exist_ok=True)
    try:
        ds = load_dataset("facebook/winoground", split="test")
    except Exception as e:
        print(f"Error loading Winoground: {e}")
        return [], None

    paths = []
    for item in tqdm(ds, desc="Preparing Images"):
        p0 = os.path.join(DATASET_CONFIG['image_root'], f"{item['id']}_0.png")
        if not os.path.exists(p0):
            item['image_0'].save(p0)
        paths.append(p0)

        p1 = os.path.join(DATASET_CONFIG['image_root'], f"{item['id']}_1.png")
        if not os.path.exists(p1):
            item['image_1'].save(p1)
        paths.append(p1)

    return paths, ds

def run_omnigen(pipe, prompt, input_images, output_path, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    final_prompt = prompt
    processed_images = []

    if input_images:
        img_tokens = ""
        for i, img_path in enumerate(input_images):
            img_tokens += f"<img><|image_{i+1}|></img> "
            processed_images.append(Image.open(img_path))
        final_prompt = img_tokens + prompt

    image = pipe(
        prompt=final_prompt,
        input_images=processed_images if processed_images else None,
        height=512,
        width=512,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        num_inference_steps=50,
        generator=generator
    ).images[0]

    image.save(output_path)

if __name__ == "__main__":
    start_time = time.time()
    seed_everything(args.seed)

    retrieval_db, ds = load_winoground_db()
    if ds is None:
        sys.exit(1)

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

    pipe, client = setup_system()
    os.makedirs(DATASET_CONFIG['output_path'], exist_ok=True)

    indices = list(range(len(ds)))
    my_indices = [i for i in indices if i % args.total_chunks == args.task_index]
    print(f"Processing {len(my_indices)} tasks.")

    global_memory = GlobalMemory()

    for i in tqdm(my_indices):
        item = ds[i]

        for suffix in ["0", "1"]:
            prompt = item[f'caption_{suffix}']
            safe_name = f"{item['id']}_{suffix}"

            log_file = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}.log")
            f_log = open(log_file, "w")
            f_log.write(f"Prompt: {prompt}\\n")

            final_success_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_FINAL.png")
            v1_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V1.png")

            if not os.path.exists(v1_path):
                run_omnigen(pipe, prompt, [], v1_path, args.seed)

            current_image = v1_path
            current_prompt = prompt
            retry_cnt = 0
            last_used_ref = None
            best_score = -1
            best_image_path = current_image

            while retry_cnt < args.max_retries:
                f_log.write(f"\\n--- Retry {retry_cnt+1} ---\\n")

                # 1. Critic (TAC)
                diagnosis = taxonomy_aware_diagnosis(
                    current_prompt, [current_image],
                    gpt_client=client, model=args.llm_model
                )

                score = diagnosis.get('final_score', 0)
                critique = diagnosis.get('critique', '')
                refined_prompt = diagnosis.get('refined_prompt', current_prompt)

                f_log.write(f"Score: {score}\\nCritique: {critique}\\n")

                if score > best_score:
                    best_score = score
                    best_image_path = current_image

                if score >= 8.0:
                    f_log.write(">> Success! Score >= 8.0\\n")
                    if not os.path.exists(final_success_path):
                        shutil.copy(current_image, final_success_path)
                    # MGR: Positive Feedback
                    if last_used_ref:
                        global_memory.add_positive(current_prompt, last_used_ref)
                    break

                # MGR: Negative Feedback
                if last_used_ref:
                    global_memory.add_negative(current_prompt, last_used_ref)

                # 2. Retrieval (MGR)
                check_token_length([refined_prompt], device="cpu", method=args.retrieval_method)

                retrieved_lists, _ = retrieve_img_per_caption(
                    [refined_prompt], retrieval_db,
                    embeddings_path=args.embeddings_path,
                    k=5, device="cuda", method=args.retrieval_method,
                    global_memory=global_memory
                )

                best_ref = None
                if retrieved_lists[0]:
                    best_ref = retrieved_lists[0][0]

                if not best_ref:
                    f_log.write("No suitable images retrieved. Proceeding without reference.\\n")
                    next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")
                    run_omnigen(pipe, refined_prompt, [], next_path, args.seed + retry_cnt + 1)
                    current_image = next_path
                    current_prompt = refined_prompt
                    retry_cnt += 1
                    continue

                f_log.write(f">> MGR Ref: {best_ref}\\n")
                last_used_ref = best_ref

                # 3. Generation with Reference
                next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")
                run_omnigen(pipe, refined_prompt, [best_ref], next_path, args.seed + retry_cnt + 1)

                current_image = next_path
                current_prompt = refined_prompt
                retry_cnt += 1

            if not os.path.exists(final_success_path):
                if os.path.exists(best_image_path):
                     shutil.copy(best_image_path, final_success_path)

            f_log.close()

    # Global Memory Training
    print("\\n============================================")
    print("All classes processed. Starting Global Memory Training...")
    try:
        trainer_memory = GlobalMemory()
        os.makedirs(os.path.join(DATASET_CONFIG['output_path'], "logs"), exist_ok=True)
        trainer_memory.train_model(epochs=20, plot_path=os.path.join(DATASET_CONFIG['output_path'], "logs", "memory_loss.png"))
    except Exception as e:
        print(f"Error during training: {e}")
    print("============================================")
