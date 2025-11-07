import os
import sys
import torch
from PIL import Image
from pathlib import Path
import argparse
import accelerate # <-- 导入 accelerate
from torchvision.transforms.functional import to_tensor # <-- 导入 to_tensor

# (OmniGen2 的导入语句将在 main 函数中)

def main(args):
    # --- 1. 设置 sys.path ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 将 OmniGen2 仓库根目录添加到 sys.path
    omnigen2_abs_path = os.path.abspath(os.path.join(script_dir, args.omnigen2_path))
    print(f"正在将 OmniGen2 仓库路径 {omnigen2_abs_path} 添加到 sys.path")
    sys.path.append(omnigen2_abs_path)

    try:
        # --- 2. 导入 OmniGen2 (已修复) ---
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
        from omnigen2.utils.img_util import create_collage
    except ImportError as e:
        print(f"错误：无法从 '{args.omnigen2_path}' 导入 OmniGen2 核心组件。")
        print(f"详细错误: {e}")
        sys.exit(1)

    # --- 3. 设置设备 (根据 app.py 修复) ---
    accelerator = accelerate.Accelerator()
    device = accelerator.device # <-- 使用 accelerator 来获取设备
    weight_dtype = torch.bfloat16
    print(f"Using device: {device}")

    # --- 4. 加载模型 (已根据 app.py 修复) ---

    # model_path 现在指向正确的 HF ID
    print(f"正在从 Hugging Face Hub ({args.model_path}) 加载 pipeline ...")

    # 1. 加载主 pipeline
    pipeline = OmniGen2Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=weight_dtype,
        trust_remote_code=True
        # (已移除 token)
    )

    # 2. 单独加载 Transformer
    print(f"正在从 {args.model_path}/transformer 加载 Transformer ...")
    pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        trust_remote_code=True
        # (已移除 token)
    )

    # 3. 将 pipeline 移动到 GPU
    pipeline = pipeline.to(device)
    print("Pipeline 加载成功。")

    # --- 5. 设置提示和生成器 ---
    negative_prompt = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # --- 6. 运行 T2I pipeline (根据 app.py 修复) ---
    print(f"正在为提示运行 T2I 生成: '{args.prompt}'...")
    results = pipeline(
        prompt=args.prompt,
        input_images=[],
        width=1024,
        height=1024,
        num_inference_steps=50,
        max_sequence_length=1024,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=1.0,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        generator=generator,
        output_type="pil",
    )

    # --- 7. 保存输出 ---
    output_image = results.images[0]
    output_image.save(args.output_path)
    print(f"图像已成功保存到: {os.path.abspath(args.output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniGen2 T2I Example Script")
    parser.add_argument("--prompt", type=str, required=True, help="用于生成的文本提示。")
    parser.add_argument("--output_path", type=str, default="output_t2i.png", help="保存输出图像的路径。")
    parser.add_argument("--omnigen2_path", type=str, default="./OmniGen2", help="克隆的 OmniGen2 仓库的路径。")
    # --- 已修复：默认值现在是正确的 HF ID ---
    parser.add_argument("--model_path", type=str, default="OmniGen2/OmniGen2", help="Hugging Face ID 或模型权重的本地路径。")
    parser.add_argument("--device_id", type=int, default=0, help="要使用的 GPU 设备 ID (会被 CUDA_VISIBLE_DEVICES 覆盖)。")
    parser.add_argument("--seed", type=int, default=0, help="用于生成的随机种子。")
    parser.add_argument("--text_guidance_scale", type=float, default=4.0, help="文本引导强度")

    args = parser.parse_args()

    # --- 自动设置 CUDA_VISIBLE_DEVICES ---
    # 这会使 'accelerator' 和 'torch.Generator' 自动选择正确的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    print(f"设置 CUDA_VISIBLE_DEVICES={args.device_id}")

    main(args)