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

# --- 1. 参数解析 ---
parser = argparse.ArgumentParser(description="OmniGenV2 + TAC + MGR (Aircraft)")

# 核心配置
parser.add_argument("--device_id", type=int, required=True, help="GPU 设备 ID")
parser.add_argument("--retrieval_device_id", type=int, default=None, help="检索 GPU 设备 ID")
parser.add_argument("--task_index", type=int, default=0)
parser.add_argument("--total_chunks", type=int, default=1)

# 路径配置
parser.add_argument("--omnigen2_path", type=str, default="./OmniGen2")
parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2")
parser.add_argument("--transformer_lora_path", type=str, default=None)
parser.add_argument("--openai_api_key", type=str, required=True)
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")

# 生成参数
parser.add_argument("--seed", type=int, default=0, help="全局随机种子")
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--text_guidance_scale", type=float, default=7.5)
parser.add_argument("--image_guidance_scale", type=float, default=1.5) # TAC 逻辑需要较高的图像引导
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/aircraft")
parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=["CLIP", "LongCLIP", "SigLIP", "SigLIP2", "BGE-VL", "Qwen2.5-VL", "Qwen3-VL"], help="检索模型")
parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter (for Qwen2.5-VL)")

args = parser.parse_args()

# 环境设置
if args.retrieval_device_id is not None and args.retrieval_device_id != args.device_id:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device_id},{args.retrieval_device_id}"
    omnigen_device = "cuda:0"
    retrieval_device = "cuda:1"
    print(f"DEBUG: 使用多 GPU。OmniGen 在 GPU {args.device_id} (内部 cuda:0)，检索在 GPU {args.retrieval_device_id} (内部 cuda:1)")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    omnigen_device = "cuda:0"
    retrieval_device = "cuda:0"
    print(f"DEBUG: 在设备 {args.device_id} 上使用单 GPU")

print(f"DEBUG: CUDA_VISIBLE_DEVICES 设置为 {os.environ['CUDA_VISIBLE_DEVICES']}")

import json
import shutil
import numpy as np
import torch
print(f"DEBUG: Torch 可见 {torch.cuda.device_count()} 个设备。当前设备: {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})")
from PIL import Image
from tqdm import tqdm
import random
import openai
import clip

# [IMPORTS] 自定义模块
from taxonomy_aware_critic import taxonomy_aware_diagnosis # TAC 逻辑
from memory_guided_retrieval import retrieve_img_per_caption
from global_memory import GlobalMemory # MGR 逻辑

# --- 2. 复现性 (固定种子) ---
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[System] 全局种子设置为: {seed}")

# --- 3. 配置 ---
DATASET_CONFIG = {
    "classes_txt": "datasets/fgvc-aircraft-2013b/data/variants.txt",
    "train_list": "datasets/fgvc-aircraft-2013b/data/images_train.txt",
    "image_root": "datasets/fgvc-aircraft-2013b/data/images",
    "output_path": "results/OmniGenV2_TAC_MGR_Aircraft"
}

# --- 4. 系统设置 ---
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
        # 修复 AttributeError
        if not hasattr(pipe.transformer, "enable_teacache"):
            pipe.transformer.enable_teacache = False
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.to("cuda")
    except ImportError as e:
        print(f"错误: 未找到 OmniGen2。详情: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    client = openai.OpenAI(
        api_key=args.openai_api_key,
        base_url="https://api.siliconflow.cn/v1/"
    )
    return pipe, client

def load_retrieval_db():
    print(f"正在加载 Aircraft 数据库...")
    paths = []
    with open(DATASET_CONFIG['train_list'], 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line: continue
            img_path = os.path.join(DATASET_CONFIG['image_root'], f"{line}.jpg")
            if os.path.exists(img_path):
                paths.append(img_path)
    print(f"已加载 {len(paths)} 张图片。")
    return paths

def run_omnigen(pipe, prompt, input_images, output_path, seed, img_guidance_scale=None, text_guidance_scale=None):
    # 确保列表格式
    if isinstance(input_images, str):
        input_images = [input_images]

    processed_imgs = []
    for img in input_images:
        try:
            if isinstance(img, str): img = Image.open(img)
            if img.mode != 'RGB': img = img.convert('RGB')
            processed_imgs.append(img)
        except: continue

    # 此次调用的确定性生成器
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

# --- 5. 主循环 ---
if __name__ == "__main__":
    import time
    start_time = time.time()

    # 1. 设置种子
    seed_everything(args.seed)

    # 2. 加载数据库并预计算 Embeddings (在加载 OmniGen 之前)
    # 这可以防止 OOM，利用空闲的 GPU 进行检索缓存。
    retrieval_db = load_retrieval_db()

    print("正在 GPU 上预计算/加载检索 Embeddings...")
    try:
        # 在完整数据库上运行一次虚拟检索以强制缓存所有图像
        # 这里使用 device="cuda" 因为 OmniGen 尚未加载。
        retrieve_img_per_caption(
            ["warmup_query"],
            retrieval_db,
            embeddings_path=args.embeddings_path,
            k=1,
            device=retrieval_device,
            method=args.retrieval_method
        )
        # 检索模型完成后清空 GPU 缓存
        torch.cuda.empty_cache()
        print("检索 Embeddings 缓存成功。")
    except Exception as e:
        print(f"Embedding 缓存期间出现警告: {e}")
        # [Fix] 如果是 CUDA 错误，必须退出，因为上下文已损坏
        if "device-side assert" in str(e) or "CUDA error" in str(e):
            print("CRITICAL: CUDA Error detected. Exiting to prevent further crashes.")
            sys.exit(1)

    # 3. 初始化 OmniGen (现在可以安全加载大模型了)
    pipe, client = setup_system()
    os.makedirs(DATASET_CONFIG['output_path'], exist_ok=True)

    # 创建日志目录
    logs_dir = os.path.join(DATASET_CONFIG['output_path'], "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # 保存运行配置
    config_path = os.path.join(logs_dir, "run_config.txt")
    with open(config_path, "w") as f:
        f.write("Run Configuration:\n")
        f.write("==================\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write(f"\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")


    # 4. 加载任务
    with open(DATASET_CONFIG['classes_txt'], 'r') as f:
        all_classes = [line.strip() for line in f.readlines() if line.strip()]

    my_classes = [c for i, c in enumerate(all_classes) if i % args.total_chunks == args.task_index]
    print(f"正在处理 {len(my_classes)} 个类别。")

    # [全局训练数据收集器]
    all_feedback_memory = []

    for class_name in tqdm(my_classes):
        safe_name = class_name.replace(" ", "_").replace("/", "-")
        prompt = f"a photo of a {class_name}"

        log_file = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}.log")
        final_success_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_FINAL.png")
        f_log = open(log_file, "w")
        f_log.write(f"Prompt: {prompt}\n")

        # 阶段 1: 初始生成
        v1_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V1.png")

        # [优化] 共享基线逻辑
        baseline_dir = "results/OmniGenV2_Baseline_Aircraft"
        baseline_v1_path = os.path.join(baseline_dir, f"{safe_name}_V1.png")

        if not os.path.exists(v1_path):
            if os.path.exists(baseline_v1_path):
                shutil.copy(baseline_v1_path, v1_path)
            else:
                run_omnigen(pipe, prompt, [], v1_path, args.seed)
                # 尝试填充基线
                try:
                    os.makedirs(baseline_dir, exist_ok=True)
                    shutil.copy(v1_path, baseline_v1_path)
                except: pass

        current_image = v1_path
        current_prompt = prompt
        retry_cnt = 0

        # [MGR 核心]: 用于重排序的全局记忆
        # 注意: GlobalMemory 目前仅支持 Qwen2.5-VL/Qwen3-VL 作为特征提取器 (MemoRAG Projector)
        memory_model_type = "Qwen2.5-VL"
        if args.retrieval_method == "Qwen3-VL":
            memory_model_type = "Qwen3-VL"

        global_memory = GlobalMemory(
            device=retrieval_device,
            embedding_model=memory_model_type,
            adapter_path=args.adapter_path
        )
        last_used_ref = None

        # [分数追踪]
        best_score = -1
        best_image_path = None

        # [知识检索] - 健全性检查
        # 每个类别检索一次规格以指导 Critic
        from taxonomy_aware_critic import generate_knowledge_specs
        try:
            reference_specs = generate_knowledge_specs(class_name, client, args.llm_model)
            f_log.write(f"Reference Specs: {reference_specs}\n")
        except Exception as e:
            f_log.write(f"Reference Specs Retrieval Failed: {e}\n")
            reference_specs = None

        while retry_cnt < args.max_retries:
            f_log.write(f"\n--- Retry {retry_cnt+1} ---\n")

            # A. TAC 诊断
            diagnosis = taxonomy_aware_diagnosis(current_prompt, [current_image], client, args.llm_model, reference_specs=reference_specs)

            score = diagnosis.get('final_score', 0)
            taxonomy_status = diagnosis.get('taxonomy_check', 'unknown')
            critique = diagnosis.get('critique', '')
            refined_prompt = diagnosis.get('refined_prompt', current_prompt)
            mgr_queries = diagnosis.get('retrieval_queries', [class_name])

            f_log.write(f"Decision: Score {score} | Taxonomy: {taxonomy_status}\nCritique: {critique}\n")
            f_log.write(f"Full Diagnosis: {json.dumps(diagnosis, indent=2)}\n")

            # [MGR 反馈循环]
            if last_used_ref is not None:
                # 如果分数 >= 6.0，我们认为参考是“有帮助/正确的概念”
                is_match = (score >= 6.0)

            # 更新最佳结果
            if score > best_score:
                best_score = score
                best_image_path = current_image

            if score >= 8.0 or (score >= 6.0 and taxonomy_status == 'correct'):
                f_log.write(f">> Success! (Score: {score}, Taxonomy: {taxonomy_status})\n")
                shutil.copy(current_image, final_success_path)
                break

            # B. 记忆引导检索 (MGR)
            # 使用来自 TAC 的特定查询

            if args.retrieval_method in ["CLIP", "SigLIP", "SigLIP2"]:
                 # 使用 LLM 生成的简明提示以避免 Token 溢出
                 query_text = diagnosis.get('concise_retrieval_prompt', f"{class_name} {class_name}")
                 # 如果 LLM 没有返回，则使用回退方案
                 if not query_text or len(query_text) < 5:
                     query_text = f"{class_name} {class_name}"
            elif args.retrieval_method in ["BGE-VL", "Qwen2.5-VL", "Qwen3-VL"]:
                 # BGE-VL/Qwen2.5-VL/Qwen3-VL: 使用完整的 prompt，不进行截断
                 query_text = f"{refined_prompt} " + " ".join(mgr_queries)
            else:
                 # 强制注入类别名称以用于长上下文模型
                 query_text = f"{class_name} {class_name}. " + " ".join(mgr_queries)
                 if len(query_text) > 300: query_text = query_text[:300]

            # [Token 长度检查]
            if args.retrieval_method not in ["BGE-VL", "Qwen2.5-VL", "Qwen3-VL"]:
                from memory_guided_retrieval import check_token_length
                check_token_length([query_text], device="cpu", method=args.retrieval_method)

            try:
                retrieved_lists, retrieved_scores = retrieve_img_per_caption(
                    [query_text], retrieval_db,
                    embeddings_path=args.embeddings_path,
                    k=50, device=retrieval_device, method=args.retrieval_method,
                    global_memory=global_memory,
                    adapter_path=args.adapter_path
                )
                candidates = retrieved_lists[0]
                candidate_scores = retrieved_scores[0]
            except RuntimeError as e:
                f_log.write(f">> Retrieval Error: {e}\n")
                candidates = []

            if not candidates:
                f_log.write(">> No references found. Proceeding without reference.\n")
                # 回退：无参考生成
                next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")

                # 使用优化后的提示词，但不使用图像
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

            # C. 生成策略
            next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")

            # [修改] 统一引导比例 (与 BC 相同)
            current_img_guidance = args.image_guidance_scale # 1.5
            current_text_guidance = args.text_guidance_scale # 7.5
            f_log.write(f">> Strategy: Unified Guidance (Image: {current_img_guidance}, Text: {current_text_guidance})\n")

            # 始终使用优化后的提示词 + 参考图
            # 用户请求: 使用 original_prompt + visual_keywords (refined_prompt)
            gen_prompt = f"{refined_prompt}. Use reference image <|image_1|>."

            run_omnigen(pipe, gen_prompt, [best_ref], next_path, args.seed + retry_cnt + 1, img_guidance_scale=current_img_guidance, text_guidance_scale=current_text_guidance)

            current_image = next_path
            # 为下一轮更新提示词 (保留完整上下文)
            current_prompt = refined_prompt
            retry_cnt += 1

        # 如果循环结束仍未成功，进行最后检查
        if not os.path.exists(final_success_path):
            f_log.write(f"\n--- Final Check (Last Generated) ---\n")
            # 评估最后生成的图像 (current_image)
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

        # 收集本次会话的记忆
        all_feedback_memory.extend(global_memory.memory)

    # --- 类别循环结束 ---
    print("\n============================================")
    print("所有类别处理完毕。开始全局记忆训练...")
    try:
        # 重新初始化以确保状态干净，并加载所有累积的记忆
        trainer_memory = GlobalMemory(
            device=retrieval_device,
            embedding_model="Qwen2.5-VL",
            adapter_path=args.adapter_path
        )
        trainer_memory.memory = all_feedback_memory # 注入收集到的记忆
        trainer_memory.train_model(epochs=20, plot_path=os.path.join(DATASET_CONFIG['output_path'], "logs", "memory_loss.png"))
    except Exception as e:
        print(f"训练期间出错: {e}")
    print("============================================")

    end_time = time.time()
    elapsed_time = end_time - start_time
    with open(os.path.join(logs_dir, "time_elapsed.txt"), "w") as f:
        f.write(f"Total execution time: {elapsed_time:.2f} seconds\n")