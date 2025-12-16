import os
import sys

import argparse
import torch
from diffusers import ZImagePipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Z-Image-Turbo Multi-GPU Inference Demo")

    parser.add_argument("--device", type=int, default=0, help="指定使用的 GPU ID (例如: 0, 1, 2, 3)")
    parser.add_argument("--prompt", type=str, default="Young Chinese woman in red Hanfu, intricate embroidery, soft lighting, 8k resolution", help="提示词")
    parser.add_argument("--output", type=str, default="result.png", help="输出文件名")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser.parse_args()

def main():
    args = parse_args()

    # 检查 CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA 设备。")

    # 检查 GPU ID 是否越界
    num_gpus = torch.cuda.device_count()
    if args.device >= num_gpus:
        raise ValueError(f"错误：设备 ID {args.device} 超出范围 (可用: {num_gpus})")

    device_str = f"cuda:{args.device}"
    gpu_name = torch.cuda.get_device_name(args.device)
    print(f"🚀 显卡: [{device_str}] {gpu_name}")

    # 加载模型
    print("⏳ 正在加载模型 (使用本地路径 ./Z-Image-Turbo)...")
    model_path = "./Z-Image-Turbo"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"本地模型路径不存在: {model_path}，请确认文件夹名称是否正确。")

    pipe = ZImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        local_files_only=True
    )

    pipe.to(device_str)

    # 尝试开启 Flash Attention
    try:
        pipe.transformer.set_attention_backend("flash")
        print("✅ Flash Attention 已启用")
    except Exception:
        print("⚠️ 未启用 Flash Attention (非致命错误)")

    print(f"⚡️ 开始生成: '{args.prompt}'")

    # 必须为 Generator 指定设备
    generator = torch.Generator(device_str).manual_seed(args.seed)

    image = pipe(
        prompt=args.prompt,
        height=512,
        width=512,
        num_inference_steps=9,  # Turbo 专属步数
        guidance_scale=0.0,     # Turbo 专属 CFG
        generator=generator,
    ).images[0]

    image.save(args.output)
    print(f"💾 完成! 图片已保存至: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()