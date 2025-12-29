'''
ZImage_TAC_MGR_Aircraft.py
=============================
Configuration:
  - Generator: Z-Image-Turbo
  - Critic: Taxonomy-Aware Critic (TAC)
  - Retrieval: Memory-Guided Retrieval (MGR)
  - Dataset: FGVC-Aircraft
'''

import argparse
import sys
import os
import time
import random
import traceback
import numpy as np
import torch
import openai
from PIL import Image
from tqdm import tqdm
import shutil
import base64

# [IMPORTS]
from taxonomy_aware_critic import taxonomy_aware_diagnosis
from memory_guided_retrieval import retrieve_img_per_caption, check_token_length
from global_memory import GlobalMemory
from diffusers import ZImagePipeline

# Proxy settings
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10000"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10000"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

parser = argparse.ArgumentParser(description="Z-Image + TAC + MGR (Aircraft)")

parser.add_argument("--device_id", type=int, required=True)
parser.add_argument("--task_index", type=int, default=0)
parser.add_argument("--total_chunks", type=int, default=1)
parser.add_argument("--model_path", type=str, default="./Z-Image-Turbo")
parser.add_argument("--openai_api_key", type=str, required=True)
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/aircraft")
parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=["CLIP", "LongCLIP", "SigLIP", "ColPali", "Hybrid", "Qwen2.5-VL"], help="Retrieval Model")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
print(f"DEBUG: CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DATASET_CONFIG = {
    "classes_txt": "datasets/fgvc-aircraft-2013b/data/variants.txt",
    "train_list": "datasets/fgvc-aircraft-2013b/data/images_train.txt",
    "image_root": "datasets/fgvc-aircraft-2013b/data/images",
    "output_path": "results/ZImage_TAC_MGR_Aircraft"
}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def fuse_prompt_with_retrieved_image(original_prompt, image_path, client, model):
    print(f"Fusion: Enhancing prompt with {image_path}...")
    try:
        base64_image = encode_image(image_path)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that optimizes prompts for image generation. You will be given an original prompt and a reference image. Your goal is to rewrite the prompt to incorporate the visual style, viewpoint, and specific details of the reference image, while maintaining the subject of the original prompt. Output ONLY the final prompt."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Original Prompt: {original_prompt}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {"type": "text", "text": "Please describe the reference image briefly and merge its visual characteristics (lighting, angle, color scheme, background) into the original prompt. The result should be a single paragraph prompt suitable for Stable Diffusion or similar models."}
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=512
        )
        new_prompt = response.choices[0].message.content.strip()
        print(f"Fusion Result: {new_prompt}")
        return new_prompt
    except Exception as e:
        print(f"Fusion failed: {e}. Using original prompt.")
        return original_prompt

def setup_system():
    print(f"Loading Z-Image from {args.model_path}...")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    pipe = ZImagePipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        local_files_only=True
    )

    pipe.to("cuda")

    try:
        pipe.transformer.set_attention_backend("flash")
        print("✅ Flash Attention Enabled")
    except Exception:
        print("⚠️ Flash Attention Not Enabled")

    client = openai.OpenAI(
        api_key=args.openai_api_key,
        base_url="https://api.siliconflow.cn/v1/"
    )
    return pipe, client

def load_retrieval_db():
    print(f"Loading Aircraft Retrieval DB...")
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

def run_zimage(pipe, prompt, input_images, output_path, seed):
    clean_prompt = prompt.replace("<|image_1|>", "").strip()

    generator = torch.Generator(device="cuda").manual_seed(seed)

    image = pipe(
        prompt=clean_prompt,
        height=1024,
        width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=generator
    ).images[0]

    image.save(output_path)

if __name__ == "__main__":
    start_time = time.time()
    seed_everything(args.seed)

    retrieval_db = load_retrieval_db()
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
        v1_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V1.png")

        if not os.path.exists(v1_path):
            run_zimage(pipe, prompt, [], v1_path, args.seed)

        current_image = v1_path
        retry_cnt = 0

        best_score = -1
        best_image_path = None

        # [MGR] Global Memory
        global_memory = GlobalMemory()
        last_used_ref = None

        # [Knowledge Retrieval] - Sanity Check
        from taxonomy_aware_critic import generate_knowledge_specs
        try:
            reference_specs = generate_knowledge_specs(class_name, client, args.llm_model)
            f_log.write(f"Reference Specs: {reference_specs}\n")
        except Exception as e:
            f_log.write(f"Reference Specs Retrieval Failed: {e}\n")
            reference_specs = None

        while retry_cnt < args.max_retries:
            f_log.write(f"\n--- Retry {retry_cnt+1} ---\n")

            # 1. Critic (TAC)
            diagnosis = taxonomy_aware_diagnosis(prompt, [current_image], client, args.llm_model, reference_specs=reference_specs)

            score = diagnosis.get('final_score', 0)
            taxonomy_status = diagnosis.get('taxonomy_check', 'unknown')
            critique = diagnosis.get('critique', '')

            f_log.write(f"Decision: Score {score} | Taxonomy: {taxonomy_status}\nCritique: {critique}\n")

            if score > best_score:
                best_score = score
                best_image_path = current_image

            if score >= 8.0 or (score >= 6.0 and taxonomy_status == 'correct'):
                f_log.write(f">> Success! (Score: {score}, Taxonomy: {taxonomy_status})\n")
                if not os.path.exists(final_success_path):
                    shutil.copy(current_image, final_success_path)
                # MGR: Positive Feedback
                if last_used_ref:
                    global_memory.add_positive(prompt, last_used_ref)
                break

            # MGR: Negative Feedback
            if last_used_ref:
                global_memory.add_negative(prompt, last_used_ref)

            # 2. Retrieval (MGR)
            # Force Class Name injection
            query_text = f"{class_name} {class_name}. {prompt}"
            check_token_length([query_text], device="cpu", method=args.retrieval_method)

            retrieved_lists, _ = retrieve_img_per_caption(
                [query_text], retrieval_db,
                embeddings_path=args.embeddings_path,
                k=5, device="cuda", method=args.retrieval_method
            )

            best_ref = None
            if retrieved_lists[0]:
                for ref in retrieved_lists[0]:
                    if not global_memory.is_excluded(prompt, ref):
                        best_ref = ref
                        break

            if not best_ref:
                f_log.write("No suitable images retrieved (all excluded or empty). Proceeding without reference.\n")
                # Fallback: Generate without reference
                next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")

                run_zimage(pipe, prompt, [], next_path, args.seed + retry_cnt + 1)

                current_image = next_path
                retry_cnt += 1
                continue

            f_log.write(f">> MGR Ref: {best_ref}\n")
            last_used_ref = best_ref

            # 3. Generation
            next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")

            # [Fusion] Use VLM to fuse prompt + retrieved image
            enhanced_prompt = fuse_prompt_with_retrieved_image(prompt, best_ref, client, args.llm_model)

            run_zimage(pipe, enhanced_prompt, [best_ref], next_path, args.seed + retry_cnt + 1)

            current_image = next_path
            retry_cnt += 1

        if not os.path.exists(final_success_path):
            f_log.write(f"\n--- Final Check (Last Generated) ---\n")
            if best_image_path and os.path.exists(best_image_path):
                 shutil.copy(best_image_path, final_success_path)
            elif os.path.exists(current_image):
                 shutil.copy(current_image, final_success_path)

        f_log.close()

    # Global Memory Training
    print("\n============================================")
    print("All classes processed. Starting Global Memory Training...")
    try:
        trainer_memory = GlobalMemory()
        os.makedirs(os.path.join(DATASET_CONFIG['output_path'], "logs"), exist_ok=True)
        trainer_memory.train_model(epochs=20, plot_path=os.path.join(DATASET_CONFIG['output_path'], "logs", "memory_loss.png"))
    except Exception as e:
        print(f"Error during training: {e}")
    print("============================================")

    end_time = time.time()
    elapsed_time = end_time - start_time
    with open(os.path.join(logs_dir, "time_elapsed.txt"), "w") as f:
        f.write(f"Total execution time: {elapsed_time:.2f} seconds\n")
