import argparse
import sys
import os
import openai
import numpy as np

from retrieval import *
from utils import *

def run_omnigen(prompt, input_images, out_path, args):
    """
    运行 OmniGen 推理生成图像。

    Args:
        prompt (str): 输入的提示词。
        input_images (list): 输入图像路径列表。
        out_path (str): 输出图像保存路径。
        args (argparse.Namespace): 命令行参数。
    """
    print("running OmniGen inference")
    # 设置设备，如果指定了 device ID 则使用对应的 CUDA 设备，否则使用默认 CUDA
    device = f"cuda:{args.device}" if int(args.device) >= 0 else "cuda"

    # 加载 OmniGen 模型
    pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1", device=device,
                                           model_cpu_offload=args.model_cpu_offload)

    # 执行图像生成
    images = pipe(prompt=prompt, input_images=input_images, height=args.height, width=args.width,
                  guidance_scale=args.guidance_scale, img_guidance_scale=args.image_guidance_scale,
                  seed=args.seed, use_input_image_size_as_output=args.use_input_image_size_as_output)

    # 保存生成的第一张图像
    images[0].save(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="imageRAG pipeline")
    # 添加命令行参数
    parser.add_argument("--omnigen_path", type=str, help="OmniGen 库的路径")
    parser.add_argument("--openai_api_key", type=str, help="OpenAI API 密钥")
    parser.add_argument("--dataset", type=str, help="数据集名称")
    parser.add_argument("--device", type=int, default=-1, help="CUDA 设备 ID，-1 表示使用默认")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="文本引导比例")
    parser.add_argument("--image_guidance_scale", type=float, default=1.6, help="图像引导比例")
    parser.add_argument("--height", type=int, default=1024, help="生成图像高度")
    parser.add_argument("--width", type=int, default=1024, help="生成图像宽度")
    parser.add_argument("--data_lim", type=int, default=-1, help="限制检索图像的数量，-1 表示不限制")
    parser.add_argument("--prompt", type=str, default="", help="输入的提示词")
    parser.add_argument("--out_name", type=str, default="out", help="输出文件名前缀")
    parser.add_argument("--out_path", type=str, default="results", help="输出目录")
    parser.add_argument("--embeddings_path", type=str, default="", help="嵌入向量路径")
    parser.add_argument("--input_images", type=str, default="", help="输入图像路径，逗号分隔")
    parser.add_argument("--mode", type=str, default="omnigen_first", choices=['omnigen_first', 'generation', 'personalization'], help="运行模式")
    parser.add_argument("--model_cpu_offload", action='store_true', help="是否将模型卸载到 CPU 以节省显存")
    parser.add_argument("--use_input_image_size_as_output", action='store_true', help="是否使用输入图像尺寸作为输出尺寸")
    parser.add_argument("--only_rephrase", action='store_true', help="是否仅重写提示词")
    parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=['CLIP', 'SigLIP', 'MoE', 'gpt_rerank'], help="检索方法")

    args = parser.parse_args()

    # 将 OmniGen 路径添加到系统路径以便导入
    sys.path.append(args.omnigen_path)
    from OmniGen import OmniGenPipeline

    # 设置 OpenAI API
    openai.api_key = args.openai_api_key
    os.environ["OPENAI_API_KEY"] = openai.api_key
    client = openai.OpenAI()

    # 创建输出目录
    os.makedirs(args.out_path, exist_ok=True)
    out_txt_file = os.path.join(args.out_path, args.out_name + ".txt")
    f = open(out_txt_file, "w")
    device = f"cuda:{args.device}" if int(args.device) >= 0 else "cuda"
    data_path = f"datasets/{args.dataset}"

    prompt_w_retreival = args.prompt

    # 获取检索图像路径列表
    retrieval_image_paths = [os.path.join(data_path, fname) for fname in os.listdir(data_path)]
    if args.data_lim != -1:
        retrieval_image_paths = retrieval_image_paths[:args.data_lim]

    # 设置嵌入向量路径
    embeddings_path = args.embeddings_path or f"datasets/embeddings/{args.dataset}"
    # 解析输入图像路径
    input_images = args.input_images.split(",") if args.input_images else []
    # 确定需要检索的概念数量 (OmniGen 最多支持 3 张图，减去已有的输入图)
    k_concepts = 3 - len(input_images) if args.mode != "personalization" else 1
    k_captions_per_concept = 1

    f.write(f"prompt: {args.prompt}\n")

    if args.mode == "omnigen_first":
        # 模式 1: 先运行 OmniGen 生成一次，然后根据结果决定是否需要 RAG
        out_name = f"{args.out_name}_no_imageRAG.png"
        out_path = os.path.join(args.out_path, out_name)
        if not os.path.exists(out_path):
            f.write(f"running OmniGen, will save results to {out_path}\n")
            run_omnigen(args.prompt, input_images, out_path, args)

        if args.only_rephrase:
            # 仅重写提示词模式
            rephrased_prompt = retrieval_caption_generation(args.prompt, input_images + [out_path],
                                                            gpt_client=client,
                                                            k_captions_per_concept=k_captions_per_concept,
                                                            only_rephrase=args.only_rephrase)
            if rephrased_prompt == True:
                # 如果返回 True，表示结果已经符合提示词，不需要修改
                f.write("result matches prompt, not running imageRAG.")
                f.close()
                exit()

            f.write(f"running OmniGen, rephrased prompt is: {rephrased_prompt}\n")
            out_name = f"{args.out_name}_rephrased.png"
            out_path = os.path.join(args.out_path, out_name)
            run_omnigen(rephrased_prompt, input_images, out_path, args)
            f.close()
            exit()
        else:
            # 正常 RAG 模式：生成检索用的 caption
            ans = retrieval_caption_generation(args.prompt,
                                               input_images + [out_path],
                                               gpt_client=client,
                                               k_captions_per_concept=k_captions_per_concept)

            if type(ans) != bool:
                captions = convert_res_to_captions(ans)
                f.write(f"captions: {captions}\n")
            else:
                # 如果返回 bool 且为 True (隐含)，表示不需要 RAG
                f.write("result matches prompt, not running imageRAG.")
                f.close()
                exit()

        omnigen_out_path = out_path

    elif args.mode == "generation":
        # 模式 2: 直接生成模式，不先运行 OmniGen
        captions = retrieval_caption_generation(args.prompt,
                                                input_images,
                                                gpt_client=client,
                                                k_captions_per_concept=k_captions_per_concept,
                                                decision=False)
        captions = convert_res_to_captions(captions)
        f.write(f"captions: {captions}\n")

    k_imgs_per_caption = 1
    # 根据 caption 检索图像
    paths = retrieve_img_per_caption(captions, retrieval_image_paths, embeddings_path=embeddings_path,
                                     k=k_imgs_per_caption, device=device, method=args.retrieval_method)
    final_paths = np.array(paths).flatten().tolist()
    j = len(input_images)
    k = 3  # OmniGen 提示词中最多可以使用 3 张图像
    paths = final_paths[:k - j]
    f.write(f"final retrieved paths: {paths}\n")
    image_paths_extended = input_images + paths

    # 构建包含检索图像的最终提示词
    examples = ", ".join([f'{captions[i]}: <img><|image_{i + j + 1}|></img>' for i in range(len(paths))])
    prompt_w_retreival = f"According to these images of {examples}, generate {args.prompt}"
    f.write(f"prompt_w_retreival: {prompt_w_retreival}\n")

    out_name = f"{args.out_name}_gs_{args.guidance_scale}_im_gs_{args.image_guidance_scale}.png"
    out_path = os.path.join(args.out_path, out_name)
    f.write(f"running OmniGen, will save result to: {out_path}\n")

    # 运行最终的 OmniGen 生成
    run_omnigen(prompt_w_retreival, image_paths_extended, out_path, args)
    f.close()
    exit()