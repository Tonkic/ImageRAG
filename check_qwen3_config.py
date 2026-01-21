
import os
import torch
from transformers import AutoConfig, AutoModelForVision2Seq

model_path = "/home/tingyu/imageRAG/Qwen3-VL-4B-Instruct"

print(f"Loading config from: {model_path}")

try:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print("\n--- Model Configuration ---")
    print(config)

    print("\n--- Key Parameters ---")
    if hasattr(config, "hidden_size"):
        print(f"config.hidden_size: {config.hidden_size}")

    if hasattr(config, "text_config"):
        print(f"config.text_config.hidden_size: {getattr(config.text_config, 'hidden_size', 'N/A')}")

    if hasattr(config, "vision_config"):
        print(f"config.vision_config.hidden_size: {getattr(config.vision_config, 'hidden_size', 'N/A')}")
        print(f"config.vision_config.embed_dim: {getattr(config.vision_config, 'embed_dim', 'N/A')}")

except Exception as e:
    print(f"Error loading config: {e}")
