import argparse
import sys
import os

# --- 1. ！！！关键修复：立即解析参数！！！ ---
# 仅导入解析参数所必需的库

parser = argparse.ArgumentParser(description="imageRAG pipeline (Batch Mode)")
# --- 核心路径和 API ---
parser.add_argument("--omnigen2_path", type=str, default="./OmniGen2", help="OmniGen2 仓库的路径")
parser.add_argument("--openai_api_key", type=str, required=True)

# --- ！！！新的：批处理和数据集参数！！！ ---
parser.add_argument("--dataset_name", type=str, required=True, choices=['aircraft', 'cub'], help="要处理的数据集的通用名称 (例如: 'aircraft', 'cub')。")
parser.add_argument("--device_id", type=int, required=True, help="要使用的 GPU 设备 ID (例如 0, 1)")
parser.add_argument("--task_index", type=int, required=True, help="任务块的索引 (例如 0, 1, 2)")
parser.add_argument("--total_chunks", type=int, default=1, help="总共的任务块数 (例如 3)")

# --- 模型加载 ---
parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2", help="模型权重的 Hugging Face ID 或本地路径")
parser.add_argument("--transformer_lora_path", type=str, default=None, help="LoRA 权重的路径")
parser.add_argument("--cpu_offload_mode", type=str, default="none", choices=['none', 'model', 'sequential'], help="OmniGen2 的 CPU Offload 模式")

# --- 生成参数 ---
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--text_guidance_scale", type=float, default=7.5)
parser.add_argument("--image_guidance_scale", type=float, default=1.6)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)

# --- RAG 和 I/O 参数 ---
parser.add_argument("--data_lim", type=int, default=-1)
parser.add_argument("--embeddings_path", type=str, default="", help="（可选）预计算 embedding 的路径。如果为空，将自动生成。")
parser.add_argument("--input_images", type=str, default="", help="（可选）为所有 prompt 添加的固定输入图像，以逗号分隔")
parser.add_argument("--mode", type=str, default="omnigen_first", choices=['omnigen_first', 'generation', 'personalization'])
parser.add_argument("--only_rephrase", action='store_true')
parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=['CLIP', 'SigLIP', 'MoE', 'gpt_rerank'])

# --- ！！！关键修复：添加 llm_model 参数！！！ ---
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct", help="用于 RAG 决策的 VLM 模型名称")

args = parser.parse_args()

# --- 2. ！！！关键修复：立即设置环境变量！！！ ---
# 必须在导入 torch、retrieval_old 或 utils 之前完成
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

# --- 3. ！！！关键修复：现在才导入所有其他库！！！ ---
import openai
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image

# (现在导入这些是安全的，因为 torch 已经被正确初始化了)
from retrieval_old import * # <-- 关键：仍然调用 retrieval_old.py
from utils_old import * # <-- 关键：调用 utils_old.py

# --------------------------------------------------
# --- 辅助函数：run_omnigen2 (不变) ---
# --------------------------------------------------
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


# --------------------------------------------------
# --- 数据库配置中心 (不变) ---
# --------------------------------------------------
DATASET_CONFIGS = {
    "aircraft": {
        "classes_txt": "datasets/fgvc-aircraft-2013b/data/variants.txt",
        "train_list": "datasets/fgvc-aircraft-2013b/data/images_train.txt",
        "image_root": "datasets/fgvc-aircraft-2013b/data/images",
        "output_path": "results/Aircraft_SmartRAG_old_retrieval" # <-- 更改了输出路径以进行A/B测试
    },
    "cub": {
        "classes_txt": "datasets/CUB_200_2011/classes.txt",
        "train_list": "datasets/CUB_200_2011/images.txt",
        "image_root": "datasets/CUB_200_2011/images",
        "output_path": "results/CUB_SmartRAG_old_retrieval" # <-- 更改了输出路径以进行A/B测试
    }
}


if __name__ == "__main__":

    # --- 1. 设置 OmniGen2 路径并导入 ---
    # (这部分是安全的，因为它不导入 torch)
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

    # --- 2. 设置 API 和设备 ---
    openai.api_key = args.openai_api_key
    os.environ["OPENAI_API_KEY"] = openai.api_key

    # --- ！！！关键修复：指定 SiliconFlow 的 base_url！！！ ---
    client = openai.OpenAI(
        api_key=args.openai_api_key,
        base_url="https://api.siliconflow.cn/v1/"
    )

    # ！！！关键：device = "cuda" 现在会正确映射到 os.environ 指定的 GPU
    device = "cuda"

    # --- 3. 加载模型 (一次性) ---
    print(f"[Device {args.device_id}] Loading OmniGen2 model (ONCE) from {args.omnigen2_model_path}...")
    if args.transformer_lora_path:
        print(f"[Device {args.device_id}] Applying LoRA weights from: {args.transformer_lora_path}")

    # (注意：这里的 'trust_remote_code=True' 可能会在日志中触发警告，但这是 OmniGen2 所需的)
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

    # (模型 CPU Offload 逻辑)
    if args.cpu_offload_mode == 'model':
        pipe.enable_model_cpu_offload()
        print(f"[Device {args.device_id}] OmniGen2 model loaded with 'model' CPU Offload enabled.")
    elif args.cpu_offload_mode == 'sequential':
        pipe.enable_sequential_cpu_offload()
        print(f"[Device {args.device_id}] OmniGen2 model loaded with 'sequential' CPU Offload (VRAM < 3GB) enabled.")
    else:
        # ！！！这就是之前失败的行 (line 174)！！！
        # 现在它将把模型正确地移动到 args.device_id 指定的 GPU
        pipe.to(device)
        print(f"[Device {args.device_id}] OmniGen2 model loaded directly onto device: {device} (Physical GPU: {args.device_id}).")

    # --- 4. ！！！新的：加载数据集和 RAG 数据库！！！ ---
    config = DATASET_CONFIGS[args.dataset_name]
    args.config_classes_txt = config["classes_txt"]
    args.config_train_list = config["train_list"]
    args.config_image_root = config["image_root"]
    args.out_path = config["output_path"]
    args.embeddings_path = args.embeddings_path or f"datasets/embeddings/{args.dataset_name}"

    os.makedirs(args.out_path, exist_ok=True)

    # --- 4a. 加载 RAG 数据库路径 (来自 smart_rag_dispatcher) ---
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

    # --- 4b. 加载并分配任务列表 (来自 smart_rag_dispatcher) ---
    print(f"[Device {args.device_id}] Loading class list from {args.config_classes_txt}...")
    all_items_to_generate = []
    try:
        with open(args.config_classes_txt) as f:
            for i, line in enumerate(f.readlines()):
                full_class_name = line.strip()
                if not full_class_name: continue

                simple_name = full_class_name # 默认为 aircraft
                if args.dataset_name == 'cub':
                    simple_name = full_class_name.split('.', 1)[-1].replace('_', ' ')
                all_items_to_generate.append((i, simple_name))
    except FileNotFoundError:
        print(f"Error: Could not find {args.config_classes_txt}.")
        sys.exit(1)

    items_for_this_gpu = []
    for i, item in enumerate(all_items_to_generate):
        if i % args.total_chunks == args.task_index:
            items_for_this_gpu.append(item)
    print(f"[Device {args.device_id}] Total classes {len(all_items_to_generate)}. This device (Task {args.task_index}) will process {len(items_for_this_gpu)}.")


    # --- 5. ！！！新的：主批处理循环！！！ ---
    for label_id, full_class_name in tqdm(items_for_this_gpu, desc=f"Generating images on Device {args.device_id} (Task {args.task_index})"):

        # --- 5a. 设置当前循环的变量 ---
        current_prompt = f"a photo of a {full_class_name}"
        safe_class_name = full_class_name.replace(' ', '_').replace('/', '_')
        current_out_name = f"{label_id:03d}_{safe_class_name}"

        out_txt_file = os.path.join(args.out_path, current_out_name + ".txt")
        f = open(out_txt_file, "w")

        f.write(f"prompt: {current_prompt}\n")

        # (从 args 加载固定的 input_images)
        input_images = args.input_images.split(",") if args.input_images else []
        k_concepts = 3 - len(input_images) if args.mode != "personalization" else 1
        k_captions_per_concept = 1


        # --- 5b. (移植自旧脚本) RAG 逻辑 ---

        if args.mode == "omnigen_first":
            out_name = f"{current_out_name}_no_imageRAG.png"
            out_path = os.path.join(args.out_path, out_name)
            if not os.path.exists(out_path):
                f.write(f"running OmniGen, will save results to {out_path}\n")
                run_omnigen2(current_prompt, input_images, out_path, args, pipe, device)

            if args.only_rephrase:
                rephrased_prompt = retrieval_caption_generation(current_prompt, input_images + [out_path],
                                                                gpt_client=client,
                                                                model=args.llm_model, # <-- 传递 model
                                                                k_captions_per_concept=k_captions_per_concept,
                                                                only_rephrase=args.only_rephrase)
                if rephrased_prompt == True:
                    f.write("result matches prompt, not running imageRAG.")
                    f.close()
                    continue

                f.write(f"running OmniGen, rephrased prompt is: {rephrased_prompt}\n")
                out_name = f"{current_out_name}_rephrased.png"
                out_path = os.path.join(args.out_path, out_name)
                run_omnigen2(rephrased_prompt, input_images, out_path, args, pipe, device)
                f.close()
                continue
            else:
                ans = retrieval_caption_generation(current_prompt,
                                                   input_images + [out_path],
                                                   gpt_client=client,
                                                   model=args.llm_model, # <-- 传递 model
                                                   k_captions_per_concept=k_captions_per_concept)

                # --------------------------------------------------
                # --- ！！！关键修复：回退到 utils_old 逻辑！！！ ---
                # --------------------------------------------------
                if type(ans) != bool:
                    captions = convert_res_to_captions(ans) # <-- 恢复调用
                    f.write(f"captions: {captions}\n")
                else: # ans is True
                    f.write("result matches prompt, not running imageRAG.")
                    f.close()
                    continue
                # --------------------------------------------------
                # (修复结束)
                # --------------------------------------------------

            omnigen_out_path = out_path

        elif args.mode == "generation":
            # --------------------------------------------------
            # --- ！！！关键修复：回退到 utils_old 逻辑！！！ ---
            # --------------------------------------------------
            captions_str = retrieval_caption_generation(current_prompt,
                                                        input_images,
                                                        gpt_client=client,
                                                        model=args.llm_model, # <-- 传递 model
                                                        k_captions_per_concept=k_captions_per_concept,
                                                        decision=False)

            captions = convert_res_to_captions(captions_str) # <-- 恢复调用
            f.write(f"captions: {captions}\n")
            # --------------------------------------------------
            # (修复结束)
            # --------------------------------------------------

        k_imgs_per_caption = 1

        # --------------------------------------------------
        # --- ！！！关键修复：使用 args.embeddings_path ！！！ ---
        # --------------------------------------------------
        paths = retrieve_img_per_caption(captions, retrieval_image_paths, embeddings_path=args.embeddings_path,
                                         k=k_imgs_per_caption, device=device, method=args.retrieval_method)
        # --------------------------------------------------
        # (修复结束)
        # --------------------------------------------------

        final_paths = np.array(paths).flatten().tolist()
        j = len(input_images)
        k = 3  # can use up to 3 images in prompt with omnigen
        paths = final_paths[:k - j]
        f.write(f"final retrieved paths: {paths}\n")
        image_paths_extended = input_images + paths

        examples = ", ".join([f'{captions[i]}: <img><|image_{i + j + 1}|></img>' for i in range(len(paths))])
        prompt_w_retreival = f"According to these images of {examples}, generate {current_prompt}"
        f.write(f"prompt_w_retreival: {prompt_w_retreival}\n")

        out_name = f"{current_out_name}_gs_{args.text_guidance_scale}_im_gs_{args.image_guidance_scale}.png"
        out_path = os.path.join(args.out_path, out_name)
        f.write(f"running OmniGen, will save result to: {out_path}\n")

        run_omnigen2(prompt_w_retreival, image_paths_extended, out_path, args, pipe, device)
        f.close()
        # <-- ！！！已移除此处的 exit()！！！ -->

    print(f"--- [Device {args.device_id}] Completed all {len(items_for_this_gpu)} tasks ---")