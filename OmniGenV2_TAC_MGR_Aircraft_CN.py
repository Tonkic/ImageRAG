'''
OmniGenV2_TAC_MGR_Aircraft.py
=============================
配置说明:
  - 生成器 (Generator): OmniGen V2
  - 评论家 (Critic): 分类感知评论家 (TAC) -> 提供细粒度的诊断
  - 检索器 (Retrieval): 记忆引导检索 (MGR) -> 动态 RAG (检索增强生成) + 排除列表机制
  - 数据集 (Dataset): FGVC-Aircraft (飞机细粒度分类数据集)

用法示例:
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

# --- 1. 参数解析 (Argument Parsing) ---
parser = argparse.ArgumentParser(description="OmniGenV2 + TAC + MGR (Aircraft)")

# 核心配置
parser.add_argument("--device_id", type=int, required=True, help="GPU 设备 ID")
parser.add_argument("--task_index", type=int, default=0, help="当前任务分片索引 (用于多卡并行)")
parser.add_argument("--total_chunks", type=int, default=1, help="总分片数")

# 路径配置
parser.add_argument("--omnigen2_path", type=str, default="./OmniGen2", help="OmniGen2 代码库路径")
parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2", help="模型权重路径")
parser.add_argument("--transformer_lora_path", type=str, default="OmniGen2-EditScore7B" if os.path.exists("OmniGen2-EditScore7B") else None, help="LoRA 权重路径")
parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI/SiliconFlow API Key")
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct", help="用于 Critic 的 VLM 模型")

# 生成参数
parser.add_argument("--seed", type=int, default=0, help="全局随机种子")
parser.add_argument("--max_retries", type=int, default=3, help="最大重试次数")
parser.add_argument("--text_guidance_scale", type=float, default=7.5, help="文本引导系数")
parser.add_argument("--image_guidance_scale", type=float, default=1.5, help="图像引导系数 (TAC 模式下通常较高)")
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/aircraft", help="检索库 Embedding 缓存路径")

args = parser.parse_args()

# 设置 CUDA 环境
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
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

# [导入] 自定义模块
from taxonomy_aware_critic import taxonomy_aware_diagnosis # TAC 逻辑 (分类感知诊断)
from memory_guided_retrieval import retrieve_img_per_caption
from global_memory import GlobalMemory # MGR 逻辑 (全局记忆)

# --- 2. 复现性设置 (固定随机种子) ---
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[System] Global seed set to: {seed}")

# --- 3. 数据集配置 ---
DATASET_CONFIG = {
    "classes_txt": "datasets/fgvc-aircraft-2013b/data/variants.txt",
    "train_list": "datasets/fgvc-aircraft-2013b/data/images_train.txt",
    "image_root": "datasets/fgvc-aircraft-2013b/data/images",
    "output_path": "results/OmniGenV2_TAC_MGR_Aircraft"
}

# --- 4. 系统初始化 ---
def setup_system():
    # 将 OmniGen2 路径加入系统路径以便导入
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(script_dir, args.omnigen2_path)))

    try:
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        # 加载 OmniGen2 Pipeline
        pipe = OmniGen2Pipeline.from_pretrained(
            args.omnigen2_model_path,
            torch_dtype=torch.float16,
            transformer_lora_path=args.transformer_lora_path,
            trust_remote_code=True,
            mllm_kwargs={"attn_implementation": "flash_attention_2"}
        )
        # 修复可能缺失的属性
        if not hasattr(pipe.transformer, "enable_teacache"):
            pipe.transformer.enable_teacache = False
        # 开启 VAE 优化以节省显存 (防止 OOM)
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.to("cuda")
    except ImportError:
        print("Error: OmniGen2 not found.")
        sys.exit(1)

    # 初始化 OpenAI 客户端 (用于调用 VLM Critic)
    client = openai.OpenAI(
        api_key=args.openai_api_key,
        base_url="https://api.siliconflow.cn/v1/"
    )
    return pipe, client

def load_retrieval_db():
    """加载检索数据库 (Aircraft 图片路径)"""
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

def run_omnigen(pipe, prompt, input_images, output_path, seed, img_guidance_scale=None):
    """运行 OmniGen 生成图像"""
    # 确保输入图片是列表格式
    if isinstance(input_images, str):
        input_images = [input_images]

    processed_imgs = []
    for img in input_images:
        try:
            if isinstance(img, str): img = Image.open(img)
            if img.mode != 'RGB': img = img.convert('RGB')
            processed_imgs.append(img)
        except: continue

    # 使用特定的 Generator 以保证确定性
    generator = torch.Generator(device="cuda").manual_seed(seed)

    if img_guidance_scale is None:
        img_guidance_scale = args.image_guidance_scale

    # 调用 Pipeline 生成
    pipe(
        prompt=prompt,
        input_images=processed_imgs,
        height=1024, width=1024,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=img_guidance_scale,
        num_inference_steps=50,
        generator=generator
    ).images[0].save(output_path)

# --- 5. 主循环 ---
if __name__ == "__main__":
    # 1. 设置随机种子
    seed_everything(args.seed)

    # 2. 加载数据库 & 预计算 Embeddings (在加载 OmniGen 之前)
    # 这样做是为了利用空闲的 GPU 显存进行检索缓存，防止 OOM。
    retrieval_db = load_retrieval_db()

    print("Pre-calculating/Loading retrieval embeddings on GPU...")
    try:
        # 运行一次虚拟检索，强制缓存所有图片的 Embedding
        # 此时 OmniGen 还没加载，所以可以用 device="cuda"
        retrieve_img_per_caption(
            ["warmup_query"],
            retrieval_db,
            embeddings_path=args.embeddings_path,
            k=1,
            device="cuda",
            method='Hybrid'
        )
        # 清理 GPU 缓存
        torch.cuda.empty_cache()
        print("Retrieval embeddings cached successfully.")
    except Exception as e:
        print(f"Warning during embedding caching: {e}")

    # 3. 初始化 OmniGen (现在可以安全加载大模型了)
    pipe, client = setup_system()
    os.makedirs(DATASET_CONFIG['output_path'], exist_ok=True)

    # 4. 加载任务列表 (类别)
    with open(DATASET_CONFIG['classes_txt'], 'r') as f:
        all_classes = [line.strip() for line in f.readlines() if line.strip()]

    # 任务分片 (多卡并行处理)
    my_classes = [c for i, c in enumerate(all_classes) if i % args.total_chunks == args.task_index]
    print(f"Processing {len(my_classes)} classes.")

    for class_name in tqdm(my_classes):
        safe_name = class_name.replace(" ", "_").replace("/", "-")
        prompt = f"a photo of a {class_name}"

        log_file = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}.log")
        f_log = open(log_file, "w")
        f_log.write(f"Prompt: {prompt}\n")

        # 阶段 1: 初始生成 (V1)
        v1_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V1.png")

        # [优化] 共享 Baseline 逻辑 (如果 Baseline 已经跑过，直接复用，节省时间)
        baseline_dir = "results/OmniGenV2_Baseline_Aircraft"
        baseline_v1_path = os.path.join(baseline_dir, f"{safe_name}_V1.png")

        if not os.path.exists(v1_path):
            if os.path.exists(baseline_v1_path):
                shutil.copy(baseline_v1_path, v1_path)
            else:
                run_omnigen(pipe, prompt, [], v1_path, args.seed)
                # 尝试保存到 Baseline 目录供后续使用
                try:
                    os.makedirs(baseline_dir, exist_ok=True)
                    shutil.copy(v1_path, baseline_v1_path)
                except: pass

        current_image = v1_path
        retry_cnt = 0

        # [MGR 核心]: 初始化全局记忆，用于重排序 (Re-ranking)
        global_memory = GlobalMemory()
        last_used_ref = None

        # [分数追踪]
        best_score = -1
        best_image_path = None

        while retry_cnt < args.max_retries:
            f_log.write(f"\n--- Retry {retry_cnt+1} ---\n")

            # A. 分类感知评论家 (TAC)
            # 调用 VLM 诊断当前图片的问题
            diagnosis = taxonomy_aware_diagnosis(prompt, [current_image], client, args.llm_model)

            status = diagnosis.get('status')
            score = diagnosis.get('score', 0)
            error_type = diagnosis.get('error_type', 'other')
            needed_modifications = diagnosis.get('needed_modifications', [])
            correct_features = diagnosis.get('correct_features', [])
            critique = diagnosis.get('critique', '')

            f_log.write(f"Decision: {status} | Score: {score} | Type: {error_type}\nCritique: {critique}\n")
            f_log.write(f"Correct: {correct_features}\nMods: {needed_modifications}\n")

            # 更新最佳结果
            if score > best_score:
                best_score = score
                best_image_path = current_image

            # 如果成功，直接结束循环
            if status == 'success':
                f_log.write(f">> Success! (Score: {score})\n")
                shutil.copy(current_image, final_success_path)
                break

            # B. 记忆引导检索 (MGR)
            # 构造查询: Prompt + 需要的修改 (目标状态)
            mod_text = ", ".join(needed_modifications)

            query_parts = [prompt]
            if needed_modifications: query_parts.append(mod_text)
            query = ". ".join(query_parts)

            # [修复] 截断查询以避免 CLIP 上下文长度溢出 (77 tokens)
            if len(query) > 300:
                query = query[:300] + "..."

            # [优化] 使用混合检索 (Hybrid Retrieval: CLIP + SigLIP) 提升 1-shot 性能
            # [修复] 使用 device="cpu" 避免 OOM。Embedding 已经在磁盘缓存好了，CPU 计算足够快。
            try:
                retrieved_lists, retrieved_scores = retrieve_img_per_caption(
                    [query], retrieval_db,
                    embeddings_path=args.embeddings_path,
                    k=50, device="cpu", method='Hybrid',
                    global_memory=global_memory
                )
                candidates = retrieved_lists[0]
                candidate_scores = retrieved_scores[0]
            except RuntimeError as e:
                f_log.write(f">> Retrieval Error (likely context length): {e}\n")
                # 降级方案: 仅使用 prompt 进行检索
                retrieved_lists, retrieved_scores = retrieve_img_per_caption(
                    [prompt], retrieval_db,
                    embeddings_path=args.embeddings_path,
                    k=50, device="cpu", method='Hybrid',
                    global_memory=global_memory
                )
                candidates = retrieved_lists[0]
                candidate_scores = retrieved_scores[0]

            # 选择最佳候选 (重排序后的 Top-1)
            if not candidates:
                f_log.write(">> Memory Exhausted (No new unique references found).\n")
                break

            best_ref = candidates[0]
            best_ref_score = candidate_scores[0]

            # 添加到全局记忆 (避免下次重复检索同一张图)
            global_memory.add(best_ref)
            last_used_ref = best_ref

            # [自适应引导系数]
            # 优化: 统一系数中心为 3.0 (BC_MGR 中的最佳实践)
            # 公式: 2.0 + (score * 4.0). 范围: [2.6, 3.4]

            # [修复] 对于 "wrong_concept" (概念错误)，放宽系数以强制改变
            f_log.write(f">> Ref: {best_ref} (Score: {best_ref_score:.4f})\n")

            # C. 动态调度与组合 (Dynamic Dispatch)
            next_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V{retry_cnt+2}.png")

            # 构造修正指令 (主动且简洁)
            # "Fix [error_type]. [Modifications]."
            instruction_parts = []
            if needed_modifications:
                instruction_parts.append(f"{mod_text}")
            else:
                instruction_parts.append(f"Fix {error_type}")

            correction_instruction = ". ".join(instruction_parts)

            # 逻辑:
            # - 对于组合/概念错误 (compositional/concept errors)，使用 Regen (重新生成)
            # - 对于其他错误，使用 Edit (编辑)
            if error_type in ["role_binding_error", "attribute_binding_error", "spatial_relation_error", "wrong_concept", "missing_object"]:
                regen_prompt = f"{prompt}. {correction_instruction}. Use <|image_1|> as a visual reference."
                f_log.write(f"Regen Prompt: {regen_prompt}\n")
                run_omnigen(pipe, regen_prompt, [best_ref], next_path, args.seed + retry_cnt + 1, img_guidance_scale=args.image_guidance_scale)
            else:
                edit_prompt = f"Edit this image to {correction_instruction}. Reference style: <|image_1|>"
                f_log.write(f"Edit Prompt: {edit_prompt}\n")
                # 对于编辑任务，通常需要更强的引导，所以稍微提高系数
                run_omnigen(pipe, edit_prompt, [current_image, best_ref], next_path, args.seed + retry_cnt + 1, img_guidance_scale=args.image_guidance_scale)

            current_image = next_path
            retry_cnt += 1

        # 最终检查: 如果循环结束仍未成功，保存最佳得分的图片
        if not os.path.exists(final_success_path):
            f_log.write(f"\n--- Final Check (Last Generated) ---\n")
            # 评估最后生成的图片
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

    # --- 类别循环结束 ---
    print("\n============================================")
    print("All classes processed. Starting Global Memory Training...")
    try:
        # 重新初始化以确保状态干净，并加载所有积累的记忆数据
        trainer_memory = GlobalMemory()
        # 训练 MLP 模型并保存 Loss 图
        trainer_memory.train_model(epochs=20, plot_path=os.path.join(DATASET_CONFIG['output_path'], "memory_loss.png"))
    except Exception as e:
        print(f"Error during training: {e}")
    print("============================================")