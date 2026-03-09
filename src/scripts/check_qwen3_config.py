
import os
import sys
import torch
import transformers

print(f"Python Executable: {sys.executable}")
print(f"Transformers Path: {transformers.__file__}")
print(f"Transformers Version: {transformers.__version__}")

# Introspect installation
all_attrs = dir(transformers)
vision_related = [a for a in all_attrs if "Vision" in a]
image_text_related = [a for a in all_attrs if "ImageText" in a] # Qwen3 might use this
print(f"Available 'Vision' attributes: {vision_related}")
print(f"Available 'ImageText' attributes: {image_text_related}")


if "AutoModelForImageTextToText" in all_attrs:
    print(">> DIAGNOSIS: AutoModelForImageTextToText is available (Recommended for Qwen3).")
elif "AutoModelForVision2Seq" in all_attrs:
    print(">> DIAGNOSIS: AutoModelForVision2Seq is detected.")
else:
    print(">> DIAGNOSIS: Neither AutoModelForImageTextToText nor AutoModelForVision2Seq found.")

try:
    from transformers import AutoConfig, AutoModelForCausalLM
except ImportError:
    pass

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

model_path = "/home/tingyu/imageRAG/Qwen3-VL-4B-Instruct"

print(f"Loading config from: {model_path}")

try:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print("\n--- Model Configuration ---")

    print("\n[Check 1] AutoModelForImageTextToText Compatibility Test (Qwen3 Recommended)")
    try:
        print("Attempting to load with AutoModelForImageTextToText...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print(">> SUCCESS: AutoModelForImageTextToText loaded the model!")
    except ImportError:
        print(">> FAIL: AutoModelForImageTextToText not found in this transformers version.")
    except Exception as e:
         print(f">> FAIL: Error loading with AutoModelForImageTextToText: {e}")

    print("\n[Check 2] AutoModelForVision2Seq Compatibility Test")
    try:
        # Just load with empty weights (meta device) if possible or cpu to just check architecture registration
        print("Attempting to load with AutoModelForVision2Seq...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print(">> SUCCESS: AutoModelForVision2Seq loaded the model!")
    except ImportError:
        print(">> FAIL: ImportError. 'transformers' might be too old or class missing.")
    except Exception as e:
        print(f">> FAIL: Error loading with AutoModelForVision2Seq: {e}")

    print("\n[Check 3] AutoModelForCausalLM Compatibility Test (Expected Failure)")
    try:
        print("Attempting to load with AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print(">> UNEXPECTED SUCCESS: AutoModelForCausalLM loaded it (should be Vision2Seq).")
    except Exception as e:
        print(f">> SUCCESS (Expected): AutoModelForCausalLM failed as expected: {e}")

except Exception as e:
    print(f"Error loading config: {e}")

model_path = "/home/tingyu/imageRAG/Qwen3-VL-4B-Instruct"

print(f"Loading config from: {model_path}")

try:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print("\n--- Model Configuration ---")
    print(config)

    print("\n[Check 1] AutoModelForVision2Seq Compatibility Test")
    try:
        # Just load with empty weights (meta device) if possible or cpu to just check architecture registration
        print("Attempting to load with AutoModelForVision2Seq...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print(">> SUCCESS: AutoModelForVision2Seq loaded the model!")
    except ImportError:
        print(">> FAIL: ImportError. 'transformers' might be too old or class missing.")
    except Exception as e:
        print(f">> FAIL: Error loading with AutoModelForVision2Seq: {e}")

    print("\n[Check 2] AutoModelForCausalLM Compatibility Test (Expected Failure)")
    try:
        print("Attempting to load with AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print(">> UNEXPECTED SUCCESS: AutoModelForCausalLM loaded it (should be Vision2Seq).")
    except Exception as e:
        print(f">> SUCCESS (Expected): AutoModelForCausalLM failed as expected: {e}")

except Exception as e:
    print(f"Error loading config: {e}")
