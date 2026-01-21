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
from datetime import datetime


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
parser.add_argument("--openai_api_key", type=str, required=False, help="Required for SiliconFlow API. If not provided, uses local model weights.")
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")

# Local Weights Config
parser.add_argument("--use_local_model_weight", action="store_true", help="Load local model weights directly (transformers)")
parser.add_argument("--local_model_weight_path", type=str, default="/home/tingyu/imageRAG/Qwen3-VL-4B-Instruct")

# 生成参数
parser.add_argument("--seed", type=int, default=0, help="全局随机种子")
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--text_guidance_scale", type=float, default=7.5)
parser.add_argument("--image_guidance_scale", type=float, default=1.5) # TAC 逻辑需要较高的图像引导
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/aircraft")
parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=["CLIP", "LongCLIP", "SigLIP", "SigLIP2", "BGE-VL", "Qwen2.5-VL", "Qwen3-VL"], help="检索模型")
parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter (for Qwen2.5-VL)")
parser.add_argument("--retrieval_datasets", nargs='+', default=['aircraft'], choices=['aircraft', 'cub', 'imagenet'], help="Datasets to use for retrieval")

args = parser.parse_args()

# 环境设置
if args.retrieval_device_id is not None and args.retrieval_device_id != args.device_id:
    # 场景 A: 双 GPU 模式 (例如 device_id=2, retrieval=3)
    # CUDA_VISIBLE_DEVICES="2,3"
    # 内部视角: cuda:0 -> 物理GPU 2 (OmniGen), cuda:1 -> 物理GPU 3 (Retrieval)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device_id},{args.retrieval_device_id}"
    omnigen_device = "cuda:0"
    retrieval_device = "cuda:1"
    print(f"DEBUG: 使用多 GPU。OmniGen 在 GPU {args.device_id} (内部 cuda:0)，检索在 GPU {args.retrieval_device_id} (内部 cuda:1)")

    # [Fix] Import Torch AFTER setting environment variables
    import torch

else:
    # 场景 B: 单 GPU 模式
    # CUDA_VISIBLE_DEVICES="2"
    # 内部视角: cuda:0 -> 物理GPU 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    omnigen_device = "cuda:0"
    retrieval_device = "cuda:0"
    print(f"DEBUG: 在设备 {args.device_id} 上使用单 GPU")

    # [Fix] Import Torch AFTER setting environment variables
    import torch

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
from rag_utils import ResourceMonitor, RUN_STATS, UsageTrackingClient, LocalQwen3VLWrapper

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
    "output_path": f"results/{_rm}/{timestamp}/OmniGenV2_TAC_MGR_Aircraft_{run_time}"
}

# --- 4. 系统设置 ---
def setup_system(shared_model=None, shared_processor=None):
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
        # 开启 TeaCache (用户请求)
        pipe.transformer.enable_teacache = True
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        # [Fix] OOM: Use enable_model_cpu_offload if configured, else specific device
        # pipe.to("cuda") # Original causing confusion with multiple GPUs
        pipe.to(omnigen_device)

    except ImportError as e:
        print(f"错误: 未找到 OmniGen2。详情: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    print("Initializing Client...")
    # [优化] 尝试共享 GlobalMemory 已加载的 Qwen3 模型 (如果存在)

    # Logic: Missing API Key -> Use Local Weights
    if not args.openai_api_key:
        print(f"  Using Local Model Weights from {args.local_model_weight_path}")
        # Pass shared_model and shared_processor if available
        client = LocalQwen3VLWrapper(
            args.local_model_weight_path,
            device_map=retrieval_device,
            shared_model=shared_model,
            shared_processor=shared_processor
        )
        # Override llm_model arg to avoid confusion
        args.llm_model = "local-qwen3-vl"
    else:
        print("  Using SiliconFlow API...")
        client = openai.OpenAI(
            api_key=args.openai_api_key,
            base_url="https://api.siliconflow.cn/v1/"
        )
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

    # 2.2 预加载 Qwen3-VL (如果需要且使用本地模型)
    # 这允许我们在 TAC Client 和 GlobalMemory/Retrieval 之间共享权重
    shared_qwen_model = None
    shared_qwen_processor = None

    need_local_qwen = (not args.openai_api_key) or (args.retrieval_method == "Qwen3-VL")
    if need_local_qwen:
        print("[System] Pre-loading Shared Qwen3-VL Model to prevent duplication...")
        from transformers import AutoProcessor, AutoModelForVision2Seq
        model_path = args.local_model_weight_path
        try:
             shared_qwen_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
             shared_qwen_model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=retrieval_device,
                trust_remote_code=True
             ).eval()
             print(f"[System] Shared Qwen3-VL loaded on {retrieval_device}")
        except Exception as e:
             print(f"[System] Failed to pre-load Qwen3-VL: {e}")

    # 3. 初始化 OmniGen (现在可以安全加载大模型了)
    pipe, client = setup_system(shared_model=shared_qwen_model, shared_processor=shared_qwen_processor)
    os.makedirs(DATASET_CONFIG['output_path'], exist_ok=True)

    # 创建日志目录
    logs_dir = os.path.join(DATASET_CONFIG['output_path'], "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Start Resource Monitor
    monitor = ResourceMonitor(interval=1.0)
    monitor.start()

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
        memory_model_type = "Qwen3-VL"
        if args.retrieval_method == "Qwen3-VL":
            memory_model_type = "Qwen3-VL"

        global_memory = GlobalMemory(
            device=retrieval_device,
            embedding_model=memory_model_type,
            adapter_path=args.adapter_path,
            external_model=shared_qwen_model,
            external_processor=shared_qwen_processor
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
                 query_text = diagnosis.get('concise_retrieval_prompt', f"{class_name}")
                 # 如果 LLM 没有返回，则使用回退方案
                 if not query_text or len(query_text) < 5:
                     query_text = f"{class_name}"
            elif args.retrieval_method in ["BGE-VL", "Qwen2.5-VL", "Qwen3-VL", "LongCLIP"]:
                 # BGE-VL/Qwen2.5-VL/Qwen3-VL/LongCLIP: 使用完整的 prompt，不进行截断
                 query_text = f"{refined_prompt} " + " ".join(mgr_queries)
            else:
                 # 强制注入类别名称以用于长上下文模型
                 query_text = f"{class_name}. " + " ".join(mgr_queries)
                 if len(query_text) > 300: query_text = query_text[:300]

            # [Token 长度检查]
            if args.retrieval_method not in ["BGE-VL", "Qwen2.5-VL", "Qwen3-VL", "LongCLIP"]:
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
            embedding_model="Qwen3-VL",
            adapter_path=args.adapter_path
        )
        trainer_memory.memory = all_feedback_memory # 注入收集到的记忆
        trainer_memory.train_model(epochs=20, plot_path=os.path.join(DATASET_CONFIG['output_path'], "logs", "memory_loss.png"))
    except Exception as e:
        print(f"训练期间出错: {e}")
    print("============================================")


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