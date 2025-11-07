import argparse
import sys
import os
import openai
import numpy as np
from tqdm import tqdm
from retrieval import *
from utils import *

# 这是一个辅助函数，它现在只负责运行模型
def run_omnigen(prompt, input_images, out_path, args, pipe):
    print("running OmniGen inference")
    # pipe 对象和 device 是从主循环传入的

    images = pipe(prompt=prompt, input_images=input_images, height=args.height, width=args.width,
                  guidance_scale=args.guidance_scale, img_guidance_scale=args.image_guidance_scale,
                  seed=args.seed, use_input_image_size_as_output=args.use_input_image_size_as_output)

    images[0].save(out_path)

# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ImageRAG Pipeline - Master Script for FGVC-Aircraft")

    # --- imageRAG_OmniGen.py 的参数 ---
    parser.add_argument("--omnigen_path", type=str)
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument("--llm_model", type=str, default="Pro/Qwen/Qwen2.5-VL-7B-Instruct", help="The name of the LLM model to use.")

    # --- 已修改：适配 datasets/ 文件夹 ---
    parser.add_argument("--dataset", type=str, default="datasets/fgvc-aircraft-2013b", help="Path to the unzipped aircraft dataset folder (e.g., 'datasets/fgvc-aircraft-2013b')")
    parser.add_argument("--out_path", type=str, default="results/Aircraft", help="Directory to save generated images and logs")
    # ------------------------------------

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.6)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--data_lim", type=int, default=-1)
    parser.add_argument("--embeddings_path", type=str, default="")
    parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=['CLIP', 'SigLIP', 'MoE', 'gpt-rerank'])
    parser.add_argument("--mode", type=str, default="omnigen_first", choices=['omnigen_first', 'generation'])
    parser.add_argument("--only_rephrase", action='store_true')
    parser.add_argument("--input_images", type=str, default="", help="Path to input images, comma separated")
    parser.add_argument("--model_cpu_offload", action='store_true')
    parser.add_argument("--use_input_image_size_as_output", action='store_true')

    # --- 并行运行的参数 ---
    parser.add_argument("--device_id", type=int, required=True, help="Actual GPU device ID to use (e.g., 0, 1, or 3)")
    parser.add_argument("--task_index", type=int, required=True, help="The index of the task chunk (e.g., 0, 1, or 2)")
    parser.add_argument("--total_chunks", type=int, default=1, help="Total number of chunks to split the job into (e.g., 3)")

    # --- 已修改：适配 datasets/ 文件夹 ---
    parser.add_argument("--classes_txt", type=str, default="datasets/fgvc-aircraft-2013b/data/variants.txt", help="Path to the variants.txt file")
    # ------------------------------------

    args = parser.parse_args()

    # --- 1. 脚本启动时的一次性设置 ---

    sys.path.append(args.omnigen_path)
    from OmniGen import OmniGenPipeline

    client = openai.OpenAI(
        api_key=args.openai_api_key,
        base_url="https://api.siliconflow.cn/v1/"
    )
    os.environ["OPENAI_API_KEY"] = args.openai_api_key

    device = f"cuda:{args.device_id}"

    os.makedirs(args.out_path, exist_ok=True)

    # --- 2. 加载模型（真正只加载一次） ---
    print(f"[Device {args.device_id}] Loading OmniGen model (ONCE) onto device: {device} ...")
    pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
    pipe.to(device)
    print(f"[Device {args.device_id}] OmniGen model loaded.")

    # --- 3. 加载 RAG 数据库路径（只需要一次） ---

    # data_path 现在会是 "datasets/fgvc-aircraft-2013b/data"
    data_path = os.path.join(args.dataset, "data")
    retrieval_image_paths = []
    train_list_file = os.path.join(data_path, "images_train.txt")

    print(f"[Device {args.device_id}] 正在从 {train_list_file} 加载 RAG 数据库...")
    try:
        with open(train_list_file, 'r') as f:
            for image_id in f.readlines():
                image_id = image_id.strip()
                if image_id:
                    image_path = os.path.join(data_path, "images", f"{image_id}.jpg")
                    if os.path.exists(image_path):
                        retrieval_image_paths.append(image_path)

        print(f"[Device {args.device_id}] 已找到 {len(retrieval_image_paths)} 张图像用于检索。")
        if not retrieval_image_paths:
            print(f"错误：在 {data_path}/images/ 中未找到图像。请检查路径。")
            sys.exit(1)

    except FileNotFoundError:
        print(f"错误：找不到 {train_list_file}。请确保 '--dataset' 参数 ('{args.dataset}') 指向了 'datasets/fgvc-aircraft-2013b' 文件夹。")
        sys.exit(1)

    if args.data_lim != -1:
        retrieval_image_paths = retrieval_image_paths[:args.data_lim]

    embeddings_path = args.embeddings_path or f"datasets/embeddings/{args.dataset.replace('/', '_')}"

    # --- 4. 加载并分配任务列表（只需要一次） ---

    print(f"[Device {args.device_id}] 正在从 {args.classes_txt} 加载类别列表...")
    all_items_to_generate = []
    try:
        with open(args.classes_txt) as f:
            for i, line in enumerate(f.readlines()):
                full_class_name = line.strip()
                if full_class_name:
                    label_id_zero_based = i
                    all_items_to_generate.append((label_id_zero_based, full_class_name))
    except FileNotFoundError:
        print(f"错误：找不到 {args.classes_txt}。")
        sys.exit(1)

    items_for_this_gpu = []
    for i, item in enumerate(all_items_to_generate):
        if i % args.total_chunks == args.task_index:
            items_for_this_gpu.append(item)

    print(f"[Device {args.device_id}] 总共 {len(all_items_to_generate)} 个类别。此设备 (Task {args.task_index}) 将处理 {len(items_for_this_gpu)} 个。")

    # --- 5. 主循环（在同一个进程中运行）---
    for label_id, full_class_name in tqdm(items_for_this_gpu, desc=f"在 Device {args.device_id} (Task {args.task_index}) 上生成图像"):

        current_prompt = f"a photo of a {full_class_name}"
        safe_class_name = full_class_name.replace(' ', '_').replace('/', '_')
        current_out_name = f"{label_id:03d}_{safe_class_name}"

        print(f"\n--- [Device {args.device_id}] G-running: {current_prompt} ---")

        out_txt_file = os.path.join(args.out_path, current_out_name + ".txt")
        f = open(out_txt_file, "w")
        f.write(f"prompt: {current_prompt}\n")

        input_images = []
        k_concepts = 3
        k_captions_per_concept = 1

        if args.mode == "omnigen_first":
            out_name = f"{current_out_name}_no_imageRAG.png"
            out_path = os.path.join(args.out_path, out_name)

            if not os.path.exists(out_path):
                f.write(f"running OmniGen, will save results to {out_path}\n")
                run_omnigen(current_prompt, input_images, out_path, args, pipe=pipe)

            if args.only_rephrase:
                rephrased_prompt = retrieval_caption_generation(current_prompt, input_images + [out_path],
                                                                    gpt_client=client, model=args.llm_model,
                                                                    k_captions_per_concept=k_captions_per_concept,
                                                                    only_rephrase=args.only_rephrase)
                if rephrased_prompt == True:
                    f.write("result matches prompt, not running imageRAG.")
                    f.close()
                    continue

                f.write(f"running OmniGen, rephrased prompt is: {rephrased_prompt}\n")
                out_name = f"{current_out_name}_rephrased.png"
                out_path = os.path.join(args.out_path, out_name)
                run_omnigen(rephrased_prompt, input_images, out_path, args, pipe=pipe)
                f.close()
                continue
            else:
                ans = retrieval_caption_generation(current_prompt, input_images + [out_path],
                                                     gpt_client=client, model=args.llm_model,
                                                     k_captions_per_concept=k_captions_per_concept)

                if type(ans) != bool:
                    captions = convert_res_to_captions(ans)
                    f.write(f"captions: {captions}\n")
                else:
                    f.write("result matches prompt, not running imageRAG.")
                    f.close()
                    continue

        elif args.mode == "generation":
            captions = retrieval_caption_generation(current_prompt, input_images,
                                                      gpt_client=client, model=args.llm_model,
                                                      k_captions_per_concept=k_captions_per_concept,
                                                      decision=False)
            captions = convert_res_to_captions(captions)
            f.write(f"captions: {captions}\n")

        k_imgs_per_caption = 1
        paths = retrieve_img_per_caption(captions, retrieval_image_paths, embeddings_path=embeddings_path,
                                         k=k_imgs_per_caption, device=device, method=args.retrieval_method)
        final_paths = np.array(paths).flatten().tolist()
        j = len(input_images)
        k = 3
        paths = final_paths[:k - j]
        f.write(f"final retrieved paths: {paths}\n")
        image_paths_extended = input_images + paths

        examples = ", ".join([f'{captions[i]}: <img><|image_{i + j + 1}|></img>' for i in range(len(paths))])
        prompt_w_retreival = f"According to these images of {examples}, generate {current_prompt}"
        f.write(f"prompt_w_retreival: {prompt_w_retreival}\n")

        out_name = f"{current_out_name}_gs_{args.guidance_scale}_im_gs_{args.image_guidance_scale}.png"
        out_path = os.path.join(args.out_path, out_name)
        f.write(f"running OmniGen, will save result to: {out_path}\n")

        run_omnigen(prompt_w_retreival, image_paths_extended, out_path, args, pipe=pipe)
        f.close()

    print(f"--- [Device {args.device_id}] 已完成所有 {len(items_for_this_gpu)} 个任务 ---")