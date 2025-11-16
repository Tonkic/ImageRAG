'''
智能 RAG 调度器 (Smart RAG Dispatcher) - [最终版 - 动态工具 - 不作弊]
=====================================
功能:
1.  --dataset_name: 允许您在 'aircraft' 和 'cub' 之间轻松切换数据库。
2.  VLM 诊断: 使用 utils.py 中的 VLM 诊断功能，返回结构化的 JSON 错误报告。
3.  工具调度 (动态提示):
    - 根据 VLM 返回的 "error_type" 动态选择“编辑”或“RAG 辅助重生成”。
4.  RAG 流程 (不作弊):
    - 阶段 1: 语义检索 (Top-K)。
    - 阶段 2: 零样本分类重排 (ZSC Reranking)。

用法:
python smart_rag_dispatcher.py \
    --device_id 1 \
    --task_index 0 \
    --total_chunks 3 \
    --image_guidance_scale 3.0 \
    --max_retries 3 \
    --dataset_name aircraft \
    --omnigen2_path ./OmniGen2 \
    --transformer_lora_path ./OmniGen2-EditScore7B \
    --openai_api_key "sk-..."
'''

import argparse
import sys
import os
import json
import shutil

# --- 1. 参数解析与环境设置 ---
parser = argparse.ArgumentParser(description="Smart RAG Dispatcher for OmniGen2 (Self-Correcting Loop Mode)")

# --- 核心配置 ---
parser.add_argument("--dataset_name", type=str, required=True, choices=['aircraft', 'cub'], help="要处理的数据集的通用名称 (例如: 'aircraft', 'cub')。")
parser.add_argument("--device_id", type=int, required=True, help="要使用的 GPU 设备 ID (例如 0, 1)")
parser.add_argument("--task_index", type=int, required=True, help="任务块的索引 (例如 0, 1, 2)")
parser.add_argument("--total_chunks", type=int, default=1, help="总共的任务块数 (例如 3)")

# --- 模型与 API 路径 ---
parser.add_argument("--omnigen2_path", type=str, default="./OmniGen2", help="OmniGen2 仓库的路径")
parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2", help="模型权重的 Hugging Face ID 或本地路径")
parser.add_argument("--transformer_lora_path", type=str, default=None, help="LoRA 权重的路径 (例如 ./OmniGen2-EditScore7B)")
parser.add_argument("--openai_api_key", type=str, required=True)
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct", help="用于 RAG 决策的 VLM 模型名称")
parser.add_argument("--cpu_offload_mode", type=str, default="none", choices=['none', 'model', 'sequential'], help="OmniGen2 的 CPU Offload 模式")

# --- 生成与 RAG 参数 ---
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--text_guidance_scale", type=float, default=7.5)
parser.add_argument("--image_guidance_scale", type=float, default=1.8, help="控制对 input_images 的遵循程度。对于 edit，推荐 3.0 左右。")
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--data_lim", type=int, default=-1)
parser.add_argument("--embeddings_path", type=str, default="", help="（可选）预计算 embedding 的路径。如果为空，将自动生成。")
parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=['CLIP', 'SigLIP'], help="用于阶段1（语义检索）的检索器。")
parser.add_argument("--max_retries", type=int, default=3, help="VLM 诊断和修正的最大尝试次数")

args = parser.parse_args()

# --- ！！！关键修复：立即设置环境变量！！！ ---
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

# --- ！！！关键修复：现在才导入所有其他库！！！ ---
import openai
import numpy as np
from tqdm import tqdm
from retrieval import * # (retrieval.py 内部会 import torch)
from utils import * # (utils.py 内部会 import torch)
from PIL import Image
import torch
import clip            # (clip 内部会 import torch)

# --------------------------------------------------
# --- 数据库配置中心 (不变) ---
# --------------------------------------------------
DATASET_CONFIGS = {
    "aircraft": {
        "classes_txt": "datasets/fgvc-aircraft-2013b/data/variants.txt",
        "train_list": "datasets/fgvc-aircraft-2013b/data/images_train.txt",
        "image_root": "datasets/fgvc-aircraft-2013b/data/images",
        "output_path": "results/Aircraft_SmartRAG"
    },
    "cub": {
        "classes_txt": "datasets/CUB_200_2011/classes.txt",
        "train_list": "datasets/CUB_200_2011/images.txt",
        "image_root": "datasets/CUB_200_2011/images",
        "output_path": "results/CUB_SmartRAG"
    }
}
config = DATASET_CONFIGS[args.dataset_name]
args.config_classes_txt = config["classes_txt"]
args.config_train_list = config["train_list"]
args.config_image_root = config["image_root"]
args.out_path = config["output_path"]
args.embeddings_path = args.embeddings_path or f"datasets/embeddings/{args.dataset_name}"
# --------------------------------------------------


# --------------------------------------------------
# --- 辅助函数：run_omnigen2 (不变) ---
# --------------------------------------------------
# (标准版：不包含 rag_images)
def run_omnigen2(prompt, images_list, out_path, args, pipe, device):
    print(f"running OmniGen2 inference... (Prompt: {prompt[:50]}...)")

    pil_images = []
    if images_list:
        for img_input in images_list:
            try:
                if isinstance(img_input, str):
                    pil_images.append(Image.open(img_input))
                elif isinstance(img_input, Image.Image):
                    pil_images.append(img_input)
                else:
                    print(f"  [Warning] 未知的图像输入类型: {type(img_input)}")
            except Exception as e:
                print(f"  [Error] 无法处理图像: {img_input}, {e}")
                return

    image_input = pil_images if pil_images else []

    # ！！！关键修复：确保 Generator 也使用正确的 device！！！
    generator = torch.Generator(device=device).manual_seed(args.seed)

    images = pipe(
        prompt=prompt,
        input_images=image_input,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        num_inference_steps=50,
        height=args.height,
        width=args.width,
        negative_prompt="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        num_images_per_prompt=1,
        generator=generator,
        output_type="pil",
    ).images

    images[0].save(out_path)
    print(f"  [Success] 已保存图像: {out_path}")

# ------------------------------
# --- 主程序入口 ---
# ------------------------------
if __name__ == "__main__":

    # --- 1. 脚本启动时的一次性设置 (不变) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    omnigen2_abs_path = os.path.abspath(os.path.join(script_dir, args.omnigen2_path))

    print(f"正在将 OmniGen2 仓库路径 {omnigen2_abs_path} 添加到 sys.path")
    sys.path.append(omnigen2_abs_path)

    try:
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
    except ImportError as e:
        print(f"错误：无法从 '{args.omnigen2_path}' 导入 OmniGen2 核心组件。")
        print(f"详细错误: {e}")
        sys.exit(1)

    client = openai.OpenAI(
        api_key=args.openai_api_key,
        base_url="https://api.siliconflow.cn/v1/"
    )
    os.environ["OPENAI_API_KEY"] = args.openai_api_key

    # ！！！关键修复：device 现在由 os.environ 控制！！！
    device = "cuda"

    os.makedirs(args.out_path, exist_ok=True)


    # --- 2. 加载模型 (不变) ---
    print(f"[Device {args.device_id}] Loading OmniGen2 model (ONCE) from {args.omnigen2_model_path}...")
    if args.transformer_lora_path:
        print(f"[Device {args.device_id}] Applying LoRA weights from: {args.transformer_lora_path}")
    pipe = OmniGen2Pipeline.from_pretrained(
        args.omnigen2_model_path,
        torch_dtype=torch.bfloat16,
        transformer_lora_path=args.transformer_lora_path,
        trust_remote_code=True
    )
    print(f"[Device {args.device_id}] Loading Transformer component separately...")
    transformer = OmniGen2Transformer2DModel.from_pretrained(
        args.omnigen2_model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    pipe.transformer = transformer

    # (现在 pipe.to(device) 会正确地移到 --device_id 指定的 GPU)
    if args.cpu_offload_mode == 'model':
        pipe.enable_model_cpu_offload()
        print(f"[Device {args.device_id}] OmniGen2 model loaded with 'model' CPU Offload enabled.")
    elif args.cpu_offload_mode == 'sequential':
        pipe.enable_sequential_cpu_offload()
        print(f"[Device {args.device_id}] OmniGen2 model loaded with 'sequential' CPU Offload (VRAM < 3GB) enabled.")
    else:
        pipe.to(device)
        print(f"[Device {args.device_id}] OmniGen2 model loaded directly onto device: {device}.")


    # --- 3. 加载 RAG 数据库路径 (不变) ---
    retrieval_image_paths = []
    print(f"[Device {args.device_id}] G-loading RAG database from {args.config_train_list}...")
    try:
        with open(args.config_train_list, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line: continue
                if args.dataset_name == 'aircraft':
                    image_path = os.path.join(args.config_image_root, f"{line}.jpg")
                elif args.dataset_name == 'cub':
                    image_filename = line.split(' ')[-1]
                    image_path = os.path.join(args.config_image_root, image_filename)
                else:
                    image_path = os.path.join(args.config_image_root, line)
                if os.path.exists(image_path):
                    retrieval_image_paths.append(image_path)
        print(f"[Device {args.device_id}] Found {len(retrieval_image_paths)} images for retrieval.")
    except FileNotFoundError:
        print(f"Error: Could not find {args.config_train_list}. Check DATASET_CONFIGS in this script.")
        sys.exit(1)
    if args.data_lim != -1:
        retrieval_image_paths = retrieval_image_paths[:args.data_lim]


    # --- 4. 加载并分配任务列表 (!!! 关键修改：用于“不作弊”的重排 !!!) ---
    print(f"[Device {args.device_id}] Loading class list from {args.config_classes_txt}...")
    all_items_to_generate = []

    # ！！！我们需要所有类别名称用于分类器！！！
    all_class_names = []

    try:
        with open(args.config_classes_txt) as f:
            for i, line in enumerate(f.readlines()):
                full_class_name = line.strip()
                if not full_class_name: continue

                simple_name = full_class_name # 默认为 aircraft
                if args.dataset_name == 'cub':
                    simple_name = full_class_name.split('.', 1)[-1].replace('_', ' ')

                all_items_to_generate.append((i, simple_name))
                all_class_names.append(simple_name) # <-- 存储所有名称

    except FileNotFoundError:
        print(f"Error: Could not find {args.config_classes_txt}.")
        sys.exit(1)

    items_for_this_gpu = []
    for i, item in enumerate(all_items_to_generate):
        if i % args.total_chunks == args.task_index:
            items_for_this_gpu.append(item)
    print(f"[Device {args.device_id}] Total classes {len(all_items_to_generate)}. This device (Task {args.task_index}) will process {len(items_for_this_gpu)}.")

    # --------------------------------------------------
    # --- ！！！关键修改：预先计算所有类别文本特征！！！ ---
    # --------------------------------------------------
    print(f"Pre-computing text features for {len(all_class_names)} classes...")
    # (加载一个临时的 CLIP 模型用于文本编码)
    clip_model_for_text, _ = clip.load("ViT-B/32", device=device)
    all_class_prompts = [f"a photo of a {name}" for name in all_class_names]

    with torch.no_grad():
        # (分批处理以防 OOM)
        all_class_text_features_list = []
        bs = 512
        for i in range(0, len(all_class_prompts), bs):
            batch_prompts = all_class_prompts[i:i+bs]
            all_class_tokens = clip.tokenize(batch_prompts).to(device)
            text_features = clip_model_for_text.encode_text(all_class_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            all_class_text_features_list.append(text_features)

        all_class_text_features = torch.cat(all_class_text_features_list, dim=0)

    del clip_model_for_text # 释放内存
    torch.cuda.empty_cache() # 清理显存
    print("Class text features pre-computed.")
    # --------------------------------------------------


    # --------------------------------------------------
    # --- 5. 主循环 (!!! 最终版：自我修正循环 !!!) ---
    # --------------------------------------------------

    PREDEFINED_ERROR_TYPES = ["success", "missing_object", "spatial_error", "count_error", "text_error", "style_error", "color_error", "wrong_concept", "other"]

    for label_id, full_class_name in tqdm(items_for_this_gpu, desc=f"Generating images on Device {args.device_id} (Task {args.task_index})"):

        current_prompt = f"a photo of a {full_class_name}"
        safe_class_name = full_class_name.replace(' ', '_').replace('/', '_')
        current_out_name = f"{label_id:03d}_{safe_class_name}"
        original_seed = args.seed

        print(f"\n--- [Device {args.device_id}] Running: {current_prompt} ---")

        out_txt_file = os.path.join(args.out_path, current_out_name + ".txt")
        f = open(out_txt_file, "w")
        f.write(f"prompt: {current_prompt}\n")

        # --- 步骤 1: 初始生成 (T2I) ---
        out_name_v1 = f"{current_out_name}_V1.png"
        out_path_v1 = os.path.join(args.out_path, out_name_v1)

        if not os.path.exists(out_path_v1):
            f.write(f"running OmniGen2 (T2I), will save results to {out_path_v1}\n")
            run_omnigen2(current_prompt, images_list=[], out_path=out_path_v1, args=args, pipe=pipe, device=device)

        # --- 步骤 2: 迭代修正循环 (Iterative Correction Loop) ---
        current_retries = 0
        current_image_to_check_path = out_path_v1
        last_generated_image_path = out_path_v1

        # --- ！！！功能 1：初始化“记忆”！！！ ---
        prompt_specific_exclusion_list = [] # 存储此 prompt 失败过的 RAG 图像
        last_used_rag_path = None            # 存储上一次循环使用的 RAG 图像
        # --- ----------------------------- ---

        while current_retries < args.max_retries:
            print(f"--- [Attempt {current_retries + 1}/{args.max_retries}] VLM 正在诊断: {current_image_to_check_path} ---")

            # 2a. VLM 诊断 (Call VLM)
            ans = retrieval_caption_generation(current_prompt, [current_image_to_check_path],
                                                 gpt_client=client, model=args.llm_model,
                                                 k_captions_per_concept=1,
                                                 decision=True)

            # 2b. 路由: 成功 (Success)
            if ans.get("status") == "success":
                f.write(f"\nVLM Decision (Attempt {current_retries + 1}): Success. Image matches prompt.\n")
                if current_retries > 0:
                    final_path = os.path.join(args.out_path, f"{current_out_name}_V_FINAL_FIXED.png")
                    shutil.copyfile(last_generated_image_path, final_path)
                    f.write(f"Successfully fixed. Copied final image to: {final_path}\n")
                break # 退出 while 循环

            # 2c. 路由: 失败 (Error)
            error_type = ans.get("error_type", "other")
            critique = ans.get("critique", "No critique provided.")
            captions = ans.get("captions")

            f.write(f"\nVLM Decision (Attempt {current_retries + 1}): {error_type}. Critique: {critique}\n")

            # --- ！！！功能 1：添加失败的 RAG 图像到“记忆”！！！ ---
            # (如果这不是第一次尝试(V1)，并且 VLM 失败了，那么上一次使用的 RAG 图像是“坏”的)
            if current_retries > 0 and last_used_rag_path:
                 if last_used_rag_path not in prompt_specific_exclusion_list:
                    prompt_specific_exclusion_list.append(last_used_rag_path)
                    f.write(f"Memory: Adding failed RAG ref {last_used_rag_path} to exclusion list for this prompt.\n")
            # --- ------------------------------------------ ---


            # 2d. 检查 VLM 错误
            if error_type not in PREDEFINED_ERROR_TYPES:
                f.write(f"WARNING: VLM returned undefined error: {error_type}. Logging to special file.\n")
                with open(os.path.join(args.out_path, "undefined_vlm_errors.txt"), "a") as err_f:
                    err_f.write(f"Prompt: {current_prompt} | Error: {error_type} | Critique: {critique}\n")
                error_type = "other"

            # 2e. 工具调度 (Tool Dispatching)

            # RAG 是所有修正工具的基础
            if not captions:
                f.write("Error: VLM failed to generate retrieval captions. Breaking loop.\n")
                break

            f.write(f"VLM Features (Captions): {captions}\n")

            # --------------------------------------------------
            # --- ！！！关键修改：不作弊的 RAG (检索 + 分类重排)！！！ ---
            # --------------------------------------------------

            # 阶段 1: 语义检索 (获取 Top K=100 个候选)
            rich_caption = [f"{current_prompt}, {captions[0]}"]
            f.write(f"Rich RAG Query: {rich_caption}\n")

            try:
                # ！！！我们假设 retrieval.py 已被修改为返回 (paths, scores, embeddings)！！！
                semantic_results = retrieve_img_per_caption(
                    rich_caption,
                    retrieval_image_paths,
                    embeddings_path=args.embeddings_path,
                    k=100,  # <-- 检索 100 个候选
                    device=device,
                    method=args.retrieval_method
                )

                if not semantic_results or not semantic_results[0] or not semantic_results[1] or not semantic_results[2]:
                     f.write("RAG (Stage 1) retrieval failed. Breaking loop.\n")
                     break

                # ！！！修复 TypeError：正确解包！！！
                semantic_paths = semantic_results[0][0]  # 第一个 caption 的 paths 列表
                semantic_scores = semantic_results[1][0] # 第一个 caption 的 scores 列表
                semantic_embeddings = semantic_results[2][0].to(device) # 第一个 caption 的 embeddings

                # --- ！！！功能 1：过滤“记忆”中的 RAG 图像！！！ ---
                if prompt_specific_exclusion_list:
                    f.write(f"Memory: Applying exclusion list: {prompt_specific_exclusion_list}\n")

                    indices_to_keep = [
                        i for i, path in enumerate(semantic_paths)
                        if path not in prompt_specific_exclusion_list
                    ]

                    if not indices_to_keep:
                        f.write("RAG (Filter) Error: Exclusion list removed all candidates. Breaking loop.\n")
                        # (重置 last_used_rag_path 以免下次循环出错)
                        last_used_rag_path = None
                        break

                    # 应用掩码 (确保使用 numpy 索引)
                    semantic_paths = np.array(semantic_paths)[indices_to_keep]
                    semantic_scores = np.array(semantic_scores)[indices_to_keep]
                    semantic_embeddings = semantic_embeddings[indices_to_keep]

                    f.write(f"RAG (Filter) Candidates remaining: {len(semantic_paths)}\n")
                # --- --------------------------------------- ---

                if semantic_embeddings.shape[0] != len(semantic_paths):
                     f.write(f"RAG (Stage 1) mismatch. Paths: {len(semantic_paths)}, Embs: {semantic_embeddings.shape[0]}. Breaking.\n")
                     break

            except Exception as e:
                 f.write(f"RAG (Stage 1) retrieval failed with error: {e}. Breaking loop.\n")
                 f.write(f"Note: This likely requires modifying retrieval.py to return (paths, scores, embeddings).\n")
                 break

            # 阶段 2: 分类重排
            target_label = full_class_name

            try:
                target_label_index = all_class_names.index(target_label)
            except ValueError:
                f.write(f"Error: Target label {target_label} not found in all_class_names. Skipping rerank.\n")
                reference_image_path = semantic_paths[0] # 回退
            else:
                # ！！！执行零样本分类！！！
                with torch.no_grad():
                    if semantic_embeddings.shape[1] != all_class_text_features.shape[1]:
                         f.write(f"Error: Embedding mismatch! Img: {semantic_embeddings.shape[1]}, Txt: {all_class_text_features.shape[1]}. Skipping rerank.\n")
                         reference_image_path = semantic_paths[0] # 回退
                    else:
                        classification_matrix = torch.matmul(semantic_embeddings, all_class_text_features.T)
                        classification_scores = classification_matrix[:, target_label_index].cpu().numpy()

                        # 阶段 3: 最终打分
                        final_scores = (semantic_scores * 0.5) + (classification_scores * 0.5)
                        final_top_index = final_scores.argsort()[-1]

                        reference_image_path = semantic_paths[final_top_index]

            # --- ！！！功能 1：记录本次使用的 RAG 图像！！！ ---
            last_used_rag_path = reference_image_path
            f.write(f"Reranked Top-1 Path: {reference_image_path} (Final Score: {final_scores[final_top_index]:.4f})\n")
            # --- ----------------------------------------- ---

            # --- RAG 修改结束 ---

            try:
                rag_ref_image_pil = Image.open(reference_image_path)
            except Exception as e:
                f.write(f"Error opening RAG PIL image: {e}. Breaking loop.\n")
                break

            # --------------------------------------------------
            # --- ！！！工具调度逻辑 (与上个版本相同)！！！ ---
            # --------------------------------------------------

            # ---- 工具组 1: RAG 编辑 (Text, Color, Other) ----
            if error_type in ["text_error", "color_error", "other"]:
                f.write(f"Tool selected: RAG (Standard ICL/Edit) for '{error_type}'.\n")

                try:
                    v_current_pil = Image.open(current_image_to_check_path)
                except Exception as e:
                    f.write(f"Error opening V-current PIL image: {e}. Breaking loop.\n")
                    break

                # 动态提示词
                if error_type == "text_error":
                    edit_prompt = f"Edit the first image: correct the text based on the prompt '{current_prompt}', using the style from the second image if necessary. Final image must show correct text."
                elif error_type == "color_error":
                    edit_prompt = f"Edit the first image: apply the correct color from the second image to the main subject. The final image should be: {current_prompt}"
                else: # other
                    edit_prompt = f"Edit the first image: use the second image as a reference to fix the errors and generate: {current_prompt}"

                f.write(f"edit_prompt (Dynamic ICL): {edit_prompt}\n")

                next_image_name = f"{current_out_name}_V{current_retries + 2}_EDITED.png"
                next_image_path = os.path.join(args.out_path, next_image_name)

                run_omnigen2(prompt=edit_prompt, images_list=[v_current_pil, rag_ref_image_pil], out_path=next_image_path, args=args, pipe=pipe, device=device)

                last_generated_image_path = next_image_path
                current_image_to_check_path = next_image_path

            # ---- 工具组 2: RAG 辅助重新生成 (Wrong Concept, Count, Spatial, Style, Missing Object) ----
            elif error_type in ["wrong_concept", "missing_object", "spatial_error", "count_error", "style_error"]:
                f.write(f"Tool selected: RAG-Assisted Regeneration for '{error_type}'.\n")

                edit_prompt = f"{current_prompt}. Use this image as a strong reference for the correct {error_type}: <|image_1|>."

                next_image_name = f"{current_out_name}_V{current_retries + 2}_RAG_REGEN.png"
                next_image_path = os.path.join(args.out_path, next_image_name)

                args.seed = original_seed + current_retries + 1 # 确保每次重试都用新 seed
                f.write(f"Running ICL (RAG-only) with new seed {args.seed} and prompt: {edit_prompt}\n")

                run_omnigen2(
                    prompt=edit_prompt,
                    images_list=[rag_ref_image_pil], # <-- 只传入 RAG 图像
                    out_path=next_image_path,
                    args=args,
                    pipe=pipe,
                    device=device
                )

                last_generated_image_path = next_image_path
                current_image_to_check_path = next_image_path

            # --------------------------------------------------
            # (修改结束)
            # --------------------------------------------------

            current_retries += 1

            if current_retries == args.max_retries:
                f.write(f"\nMax retries ({args.max_retries}) reached. Stopping correction loop.\n")
                f.write(f"Final failed image: {last_generated_image_path}\n")

    f.close()
    args.seed = original_seed # 恢复原始 seed 以便下一个 prompt

print(f"--- [Device {args.device_id}] Completed all {len(items_for_this_gpu)} tasks ---")