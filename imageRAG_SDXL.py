import argparse
import os
from PIL import Image
import numpy as np
import openai
import torch
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from transformers import CLIPVisionModelWithProjection
import base64
import io

from utils import *
from retrieval import *

# 辅助函数：将图片编码为Base64
def encode_image_to_base64(image_path):
    """Encodes an image file to a Base64 string."""
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_byte = buffered.getvalue()
        return base64.b64encode(img_byte).decode('utf-8')

# 通用的文本/视觉生成函数
def generate_caption_or_rephrase(client, model_name, prompt, image_paths=[], only_rephrase=False, decision=True):
    """
    Generates a caption or rephrases a prompt using a specified LLM service.
    Supports both text-only and vision-language models.
    """
    content = [{"type": "text", "text": prompt}]
    if image_paths:
        for path in image_paths:
            base64_image = encode_image_to_base64(path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

    messages = [{"role": "user", "content": content}]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=300
        )
        result = response.choices[0].message.content.strip()

        if decision:
            if result.lower() in prompt.lower() or prompt.lower() in result.lower():
                 return False

        return result
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="imageRAG pipeline")

    # --- API 和模型配置参数  ---
    # MODIFIED: 保留参数名，但明确其用途是 SiliconFlow Key。可以从环境变量 SILICONFLOW_API_KEY 读取
    parser.add_argument("--openai_api_key", type=str, default=os.getenv("SILICONFLOW_API_KEY"), help="Your SiliconFlow API key (argument name kept for consistency).")
    # MODIFIED: 重命名为 --llm_model 并设为必填
    parser.add_argument("--llm_model", type=str, required=True, help="Model name from SiliconFlow to use (e.g., 'Qwen/Qwen3-VL-30B-A3B-Instruct').")

    # --- 原始参数 ---
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--ip_scale", type=float, default=0.5)
    parser.add_argument("--data_lim", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--out_name", type=str, default="out")
    parser.add_argument("--out_path", type=str, default="results")
    parser.add_argument("--embeddings_path", type=str, default="")
    parser.add_argument("--mode", type=str, default="sd_first", choices=['sd_first', 'generation'])
    parser.add_argument("--only_rephrase", action='store_true')
    parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=['CLIP', 'SigLIP', 'SigLIP2', 'MoE', 'gpt_rerank'])

    args = parser.parse_args()

    # --- 初始化LLM客户端 (已修改) ---
    # MODIFIED: 客户端固定指向 SiliconFlow
    if not args.openai_api_key:
        raise ValueError("SiliconFlow API key is required. Use --openai_api_key or set the SILICONFLOW_API_KEY environment variable.")

    llm_client = openai.OpenAI(
        api_key=args.openai_api_key,
        base_url="https://api.siliconflow.cn/v1"
    )

    # --- 脚本主体逻辑 (与之前版本基本保持一致) ---
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

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
        cache_dir=args.hf_cache_dir
    )

    pipe_clean = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        cache_dir=args.hf_cache_dir
    ).to(device)

    generator1 = torch.Generator(device="cuda").manual_seed(args.seed)
    pipe_ip = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        cache_dir=args.hf_cache_dir
    ).to(device)

    pipe_ip.load_ip_adapter("h94/IP-Adapter",
                            subfolder="sdxl_models",
                            weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                            cache_dir=args.hf_cache_dir)

    pipe_ip.set_ip_adapter_scale(args.ip_scale)
    generator2 = torch.Generator(device=device).manual_seed(args.seed)

    sd_first = args.mode == "sd_first"

    if sd_first:
        cur_out_path = os.path.join(args.out_path, f"{args.out_name}_no_imageRAG.png")
        if not os.path.exists(cur_out_path):
            out_image = pipe_clean(
                prompt=args.prompt,
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                num_inference_steps=50,
                generator=generator1,
            ).images[0]
            out_image.save(cur_out_path)

        # MODIFIED: 使用 --llm_model
        ans = generate_caption_or_rephrase(llm_client,
                                           args.llm_model,
                                           args.prompt,
                                           image_paths=[cur_out_path],
                                           only_rephrase=args.only_rephrase)
        if type(ans) != bool:
            if args.only_rephrase:
                print(f"running SDXL, rephrased prompt is: {ans}\n")
                cur_out_path = os.path.join(args.out_path, f"{args.out_name}_rephrased.png")
                out_image = pipe_clean(
                    prompt=ans,
                    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                    num_inference_steps=50,
                    generator=generator1,
                ).images[0]
                out_image.save(cur_out_path)
                exit()

            caption = ans
            caption = convert_res_to_captions(caption)[0]
            print(f"caption: {caption}\n")
        else:
            print(f"prompt: {args.prompt}")
            print("result matches prompt, not running imageRAG.")
            exit()
    else:
        # MODIFIED: 使用 --llm_model
        caption = generate_caption_or_rephrase(llm_client,
                                               args.llm_model,
                                               args.prompt,
                                               image_paths=[],
                                               decision=False)
        caption = convert_res_to_captions(caption)[0]
        f.write(f"captions: {caption}\n")

    paths = retrieve_img_per_caption([caption], retrieval_image_paths, embeddings_path=embeddings_path,
                                     k=1, device=device, method=args.retrieval_method)
    image_path = np.array(paths).flatten()[0]
    print("ref path:", image_path)

    new_prompt = f"According to this image of {caption}, generate {args.prompt}"
    image = Image.open(image_path)

    out_image = pipe_ip(
        prompt=new_prompt,
        ip_adapter_image=image,
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        num_inference_steps=50,
        generator=generator2,
    ).images[0]

    cur_out_path = os.path.join(args.out_path, f"{args.out_name}.png")
    out_image.save(cur_out_path)