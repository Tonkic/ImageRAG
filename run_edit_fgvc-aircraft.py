'''
python run_edit_fgvc-aircraft.py \
--device_id 0 \
--task_index 0 \
--total_chunks 3 \
--omnigen2_path ./OmniGen2 \
--transformer_lora_path ./OmniGen2-EditScore7B \
--openai_api_key sk-dxbmolkckwxjqepbnfxgjtgonbtkixrtmmmiasjpadvjfgur
'''


import argparse
import sys
import os

# ！！关键修复！！
# 我们必须在导入 openai, torch, 等库之前
# 解析参数并设置 CUDA_VISIBLE_DEVICES

# 1. 解析命令行参数
parser = argparse.ArgumentParser(description="OmniGen2 RAG+Edit Pipeline for FGVC-Aircraft")
# ... (所有 parser.add_argument 调用保持不变) ...
# --- OmniGen2 和 API 的参数 ---
parser.add_argument("--omnigen2_path", type=str, default="./OmniGen2", help="Path to the OmniGen2 checkout/folder")
parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2", help="Hugging Face ID 或模型权重的本地路径。")
parser.add_argument("--transformer_lora_path", type=str, default=None, help="Path to the LoRA weights (e.g., ./OmniGen2-EditScore7B)")
parser.add_argument("--openai_api_key", type=str)
parser.add_argument("--llm_model", type=str, default="Pro/Qwen/Qwen2.5-VL-7B-Instruct", help="The name of the VLM model to use for RAG decision.")
parser.add_argument("--cpu_offload_mode", type=str, default="model", choices=['none', 'model', 'sequential'],
                    help="CPU Offload mode for OmniGen2. 'model' reduces VRAM by 50%%, 'sequential' reduces to <3GB.")
# --- 数据集和路径参数 ---
parser.add_argument("--dataset", type=str, default="datasets/fgvc-aircraft-2013b", help="Path to the unzipped aircraft dataset folder")
parser.add_argument("--out_path", type=str, default="results/Aircraft_OmniGen2_LoRA", help="Directory to save generated images and logs")
parser.add_argument("--classes_txt", type=str, default="datasets/fgvc-aircraft-2013b/data/variants.txt", help="Path to the variants.txt file")
# --- 生成和 RAG 参数 ---
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--text_guidance_scale", type=float, default=7.5, help="CFG for text (default in OmniGen2 is 7.5)")
parser.add_argument("--image_guidance_scale", type=float, default=1.8, help="Guidance for image/edit (default in OmniGen2 is ~1.2-2.0)")
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--data_lim", type=int, default=-1)
parser.add_argument("--embeddings_path", type=str, default="")
parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=['CLIP', 'SigLIP', 'MoE', 'gpt-rerank'])
# --- 并行运行的参数 ---
parser.add_argument("--device_id", type=int, required=True, help="Actual GPU device ID to use (e.g., 0, 1, or 3)")
parser.add_argument("--task_index", type=int, required=True, help="The index of the task chunk (e.g., 0, 1, or 2)")
parser.add_argument("--total_chunks", type=int, default=1, help="Total number of chunks to split the job into (e.g., 3)")

args = parser.parse_args()

# --- 2. ！！！关键修复：立即设置环境变量！！！ ---
# 这会告诉 accelerate (被 offload 使用) 只能看到这一个 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
# ----------------------------------------------------


# --- 3. 现在再导入其他所有库 ---
import openai
import numpy as np
from tqdm import tqdm
from retrieval import *
from utils import *
from PIL import Image
import torch

# (run_edit_fgvc-aircraft.py, 约 86 行)
def run_omnigen2(prompt, images_list, out_path, args, pipe, device, rag_images=None): # <-- 1. 添加 rag_images=None
    print(f"running OmniGen2 inference... (Prompt: {prompt[:50]}...)")

    pil_images = []
    if images_list:
        for img_input in images_list: # <-- 重命名变量
            try:
                # 检查是路径 (str) 还是已加载的 PIL.Image 对象
                if isinstance(img_input, str):
                    pil_images.append(Image.open(img_input))
                elif isinstance(img_input, Image.Image):
                    pil_images.append(img_input) # <-- 直接添加 PIL 对象
                else:
                    print(f"  [Warning] 未知的图像输入类型: {type(img_input)}")
            except Exception as e:
                print(f"  [Error] 无法处理图像: {img_input}, {e}")
                return

    image_input = pil_images if pil_images else []

    # 'device' 现在会被 accelerate 自动管理 (因为设置了 CUDA_VISIBLE_DEVICES)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    images = pipe(
        prompt=prompt,
        input_images=image_input,
        rag_images=rag_images,       # <-- 2. 在此处传递新参数
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        # ... (其他参数保持不变) ...
    ).images

    images[0].save(out_path)
    print(f"  [Success] 已保存图像: {out_path}")

# ------------------------------

# 主程序入口
if __name__ == "__main__":

    # (Args 已经被解析过了)

    # --- 1. 脚本启动时的一次性设置 ---

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

    # --- 已修改：'device' 现在由 accelerate/CUDA_VISIBLE_DEVICES 自动管理 ---
    device = "cuda" # (或者 torch.device("cuda"))
    # -----------------------------------------------------------------

    os.makedirs(args.out_path, exist_ok=True)

    # --- 2. 加载模型（只加载 OmniGen2） ---

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

    # --- 已修改：offload 现在会自动使用正确的 (可见的) GPU ---
    if args.cpu_offload_mode == 'model':
        pipe.enable_model_cpu_offload()
        print(f"[Device {args.device_id}] OmniGen2 model loaded with 'model' CPU Offload enabled.")
    elif args.cpu_offload_mode == 'sequential':
        pipe.enable_sequential_cpu_offload()
        print(f"[Device {args.device_id}] OmniGen2 model loaded with 'sequential' CPU Offload (VRAM < 3GB) enabled.")
    else:
        pipe.to(device)
        print(f"[Device {args.device_id}] OmniGen2 model loaded directly onto device: {device}.")
    # ----------------------------------------------------------

    # --- 3. 加载 RAG 数据库路径 ---
    data_path = os.path.join(args.dataset, "data")
    retrieval_image_paths = []
    train_list_file = os.path.join(data_path, "images_train.txt")

    print(f"[Device {args.device_id}] G-loading RAG database from {train_list_file}...")
    try:
        with open(train_list_file, 'r') as f:
            for image_id in f.readlines():
                image_id = image_id.strip()
                if image_id:
                    image_path = os.path.join(data_path, "images", f"{image_id}.jpg")
                    if os.path.exists(image_path):
                        retrieval_image_paths.append(image_path)
        print(f"[Device {args.device_id}] Found {len(retrieval_image_paths)} images for retrieval.")
    except FileNotFoundError:
        print(f"Error: Could not find {train_list_file}. Ensure '--dataset' arg ('{args.dataset}') points to 'datasets/fgvc-aircraft-2013b' folder.")
        sys.exit(1)

    if args.data_lim != -1:
        retrieval_image_paths = retrieval_image_paths[:args.data_lim]
    embeddings_path = args.embeddings_path or f"datasets/embeddings/{args.dataset.replace('/', '_')}"

    # --- 4. 加载并分配任务列表 ---
    print(f"[Device {args.device_id}] Loading class list from {args.classes_txt}...")
    all_items_to_generate = []
    try:
        with open(args.classes_txt) as f:
            for i, line in enumerate(f.readlines()):
                full_class_name = line.strip()
                if full_class_name:
                    all_items_to_generate.append((i, full_class_name))
    except FileNotFoundError:
        print(f"Error: Could not find {args.classes_txt}.")
        sys.exit(1)

    items_for_this_gpu = []
    for i, item in enumerate(all_items_to_generate):
        if i % args.total_chunks == args.task_index:
            items_for_this_gpu.append(item)
    print(f"[Device {args.device_id}] Total classes {len(all_items_to_generate)}. This device (Task {args.task_index}) will process {len(items_for_this_gpu)}.")

    # --- 5. 主循环（在同一个进程中运行）---
    for label_id, full_class_name in tqdm(items_for_this_gpu, desc=f"Generating images on Device {args.device_id} (Task {args.task_index})"):

        current_prompt = f"a photo of a {full_class_name}"
        safe_class_name = full_class_name.replace(' ', '_').replace('/', '_')
        current_out_name = f"{label_id:03d}_{safe_class_name}"

        print(f"\n--- [Device {args.device_id}] Running: {current_prompt} ---")

        out_txt_file = os.path.join(args.out_path, current_out_name + ".txt")
        f = open(out_txt_file, "w")
        f.write(f"prompt: {current_prompt}\n")

        k_captions_per_concept = 1

        # --- 步骤 1: 初始生成 (T2I) ---
        out_name_no_rag = f"{current_out_name}_no_imageRAG.png"
        out_path_no_rag = os.path.join(args.out_path, out_name_no_rag)

        if not os.path.exists(out_path_no_rag):
            f.write(f"running OmniGen2 (T2I), will save results to {out_path_no_rag}\n")
            run_omnigen2(current_prompt, images_list=[], out_path=out_path_no_rag, args=args, pipe=pipe, device=device)

        # --- 步骤 2: VLM 决策 ---
        ans = retrieval_caption_generation(current_prompt, [out_path_no_rag],
                                             gpt_client=client, model=args.llm_model,
                                             k_captions_per_concept=k_captions_per_concept)

        if type(ans) == bool and ans == True:
            f.write("result matches prompt, not running RAG edit.")
            f.close()
            continue

        captions = convert_res_to_captions(ans)
        f.write(f"captions: {captions}\n")

        # --- 步骤 3: RAG 检索 ---
        k_imgs_per_caption = 1
        paths = retrieve_img_per_caption(captions, retrieval_image_paths, embeddings_path=embeddings_path,
                                         k=k_imgs_per_caption, device=device, method=args.retrieval_method)

        if not paths or not paths[0]:
            f.write("RAG retrieval failed to find images. Skipping edit.")
            f.close()
            continue

        reference_image_path = paths[0][0]
        f.write(f"final retrieved path (as reference): {reference_image_path}\n")

        # --- 步骤 4: RAG 编辑 (Embedding Injection) ---

        # 1. 加载 V1 图像和 RAG 参考图像作为 PIL 对象
        try:
            v1_image_pil = Image.open(out_path_no_rag)
            rag_ref_image_pil = Image.open(reference_image_path)
        except Exception as e:
            f.write(f"Error opening PIL images for RAG edit: {e}\n")
            f.close()
            continue

        # 2. 使用简单的 T2I 提示（模型现在通过 rag_images 获取概念）
        edit_prompt = f"a photo of a {full_class_name}"
        f.write(f"edit_prompt (Injection): {edit_prompt}\n")

        out_name_rag = f"{current_out_name}_EDITED.png"
        out_path_final = os.path.join(args.out_path, out_name_rag)
        f.write(f"running OmniGen2 (ICL + RAG Injection), will save result to: {out_path_final}\n")

        # 3. 使用新参数调用函数
        run_omnigen2(
            prompt=edit_prompt,
            images_list=[v1_image_pil],      # V1 图像用于 ICL (原始 input_images)
            rag_images=[rag_ref_image_pil],  # RAG 图像用于新注入 (新 rag_images)
            out_path=out_path_final,
            args=args,
            pipe=pipe,
            device=device
        )
        f.close()
    print(f"--- [Device {args.device_id}] Completed all {len(items_for_this_gpu)} tasks ---")