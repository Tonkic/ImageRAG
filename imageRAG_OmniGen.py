import argparse
import sys
import os
import openai
import numpy as np

from retrieval import *
from utils import *

def run_omnigen(prompt, input_images, out_path, args):
    print("running OmniGen inference")
    device = f"cuda:{args.device}" if int(args.device) >= 0 else "cuda"

    # 从 from_pretrained 调用中移除所有不支持的参数
    # 只需要保留模型名称即可
    pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")

    # 将加载好的 pipeline 移动到 GPU
    pipe.to(device)

    # 后面的代码保持不变
    images = pipe(prompt=prompt, input_images=input_images, height=args.height, width=args.width,
                  guidance_scale=args.guidance_scale, img_guidance_scale=args.image_guidance_scale,
                  seed=args.seed, use_input_image_size_as_output=args.use_input_image_size_as_output)

    images[0].save(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="imageRAG pipeline")
    parser.add_argument("--omnigen_path", type=str)
    parser.add_argument("--openai_api_key", type=str)
    # ADDED: New argument for the custom LLM model name
    parser.add_argument("--llm_model", type=str, default="Pro/Qwen/Qwen2.5-VL-7B-Instruct", help="The name of the LLM model to use.")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.6)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--data_lim", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--out_name", type=str, default="out")
    parser.add_argument("--out_path", type=str, default="results")
    parser.add_argument("--embeddings_path", type=str, default="")
    parser.add_argument("--input_images", type=str, default="")
    parser.add_argument("--mode", type=str, default="omnigen_first", choices=['omnigen_first', 'generation', 'personalization'])
    parser.add_argument("--model_cpu_offload", action='store_true')
    parser.add_argument("--use_input_image_size_as_output", action='store_true')
    parser.add_argument("--only_rephrase", action='store_true')
    parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=['CLIP', 'SigLIP', 'MoE', 'gpt_rerank'])

    args = parser.parse_args()

    sys.path.append(args.omnigen_path)
    from OmniGen import OmniGenPipeline

    # CHANGED: Initialize the client with the custom base URL from siliconflow
    # The URL for the base_url should not include the /chat/completions part.
    client = openai.OpenAI(
        api_key=args.openai_api_key,
        base_url="https://api.siliconflow.cn/v1/"
    )
    os.environ["OPENAI_API_KEY"] = args.openai_api_key # Some libraries might still read from env

    os.makedirs(args.out_path, exist_ok=True)
    out_txt_file = os.path.join(args.out_path, args.out_name + ".txt")
    f = open(out_txt_file, "w")
    device = f"cuda:{args.device}" if int(args.device) >= 0 else "cuda"
    data_path = f"datasets/{args.dataset}"

    prompt_w_retreival = args.prompt

    retrieval_image_paths = [os.path.join(data_path, fname) for fname in os.listdir(data_path)]
    if args.data_lim != -1:
        retrieval_image_paths = retrieval_image_paths[:args.data_lim]

    embeddings_path = args.embeddings_path or f"datasets/embeddings/{args.dataset}"
    input_images = args.input_images.split(",") if args.input_images else []
    k_concepts = 3 - len(input_images) if args.mode != "personalization" else 1
    k_captions_per_concept = 1

    f.write(f"prompt: {args.prompt}\n")

    if args.mode == "omnigen_first":
        out_name = f"{args.out_name}_no_imageRAG.png"
        out_path = os.path.join(args.out_path, out_name)
        if not os.path.exists(out_path):
            f.write(f"running OmniGen, will save results to {out_path}\n")
            run_omnigen(args.prompt, input_images, out_path, args)

        if args.only_rephrase:
            # CHANGED: Pass the model name to the function
            rephrased_prompt = retrieval_caption_generation(args.prompt, input_images + [out_path],
                                                                gpt_client=client,
                                                                model=args.llm_model,
                                                                k_captions_per_concept=k_captions_per_concept,
                                                                only_rephrase=args.only_rephrase)
            if rephrased_prompt == True:
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
            # CHANGED: Pass the model name to the function
            ans = retrieval_caption_generation(args.prompt,
                                                 input_images + [out_path],
                                                 gpt_client=client,
                                                 model=args.llm_model,
                                                 k_captions_per_concept=k_captions_per_concept)

            if type(ans) != bool:
                captions = convert_res_to_captions(ans)
                f.write(f"captions: {captions}\n")
            else:
                f.write("result matches prompt, not running imageRAG.")
                f.close()
                exit()

        omnigen_out_path = out_path

    elif args.mode == "generation":
        # CHANGED: Pass the model name to the function
        captions = retrieval_caption_generation(args.prompt,
                                                  input_images,
                                                  gpt_client=client,
                                                  model=args.llm_model,
                                                  k_captions_per_concept=k_captions_per_concept,
                                                  decision=False)
        captions = convert_res_to_captions(captions)
        f.write(f"captions: {captions}\n")

    k_imgs_per_caption = 1
    paths = retrieve_img_per_caption(captions, retrieval_image_paths, embeddings_path=embeddings_path,
                                     k=k_imgs_per_caption, device=device, method=args.retrieval_method)
    final_paths = np.array(paths).flatten().tolist()
    j = len(input_images)
    k = 3  # can use up to 3 images in prompt with omnigen
    paths = final_paths[:k - j]
    f.write(f"final retrieved paths: {paths}\n")
    image_paths_extended = input_images + paths

    examples = ", ".join([f'{captions[i]}: <img><|image_{i + j + 1}|></img>' for i in range(len(paths))])
    prompt_w_retreival = f"According to these images of {examples}, generate {args.prompt}"
    f.write(f"prompt_w_retreival: {prompt_w_retreival}\n")

    out_name = f"{args.out_name}_gs_{args.guidance_scale}_im_gs_{args.image_guidance_scale}.png"
    out_path = os.path.join(args.out_path, out_name)
    f.write(f"running OmniGen, will save result to: {out_path}\n")

    run_omnigen(prompt_w_retreival, image_paths_extended, out_path, args)
    f.close()
    exit()