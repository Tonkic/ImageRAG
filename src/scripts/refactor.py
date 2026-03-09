import os
import re

file_path = r"e:\ImageRAG-main\src\experiments\OmniGenV2_TAC_DINO_Importance_Aircraft.py"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. FIX IMPORTS: We need to pull `torch`, `clip`, etc. DOWN below `CUDA_VISIBLE_DEVICES` logic.
import_pattern = r"""# ==============================================================================
# Imports
# ==============================================================================
(.*?)
# ==============================================================================
# Argument Parsing
# ==============================================================================
"""

def replace_imports(match):
    original_imports = match.group(1)
    # Keep only sys, os, argparse, datetime at top
    new_top = """from datetime import datetime\nimport argparse\nimport sys\nimport os\n\n# Wait to import torch until CUDA is set\n"""
    return "# ==============================================================================\n# Imports\n# ==============================================================================\n" + new_top + "\n# ==============================================================================\n# Argument Parsing\n# ==============================================================================\n"

text = re.sub(import_pattern, replace_imports, text, flags=re.DOTALL)

# Now inject the heavy imports AFTER the Torch sees X devices print statement
device_setup_pattern = r"""print\(f"\[Config\] Torch sees \{torch\.cuda\.device_count\(\)\} device\(s\)"\)"""

heavy_imports = """
# ==============================================================================
# Heavy Imports (Post-CUDA setup)
# ==============================================================================
import gc
import json
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import random
from PIL import Image
from tqdm import tqdm
import openai
import clip
import time
import io
import base64

print(f"[Config] Torch sees {torch.cuda.device_count()} device(s)")
"""
text = re.sub(device_setup_pattern, heavy_imports.replace('\\', '\\\\'), text)


# 2. FIX LOAD_DB
load_db_pattern = r"""def load_db\(use_nobg=False\):.*?return all_paths"""

new_load_db = """def load_db(use_nobg=False):
    \"\"\"Load retrieval database images. Returns a dict of ds_name -> paths instead of flat list.\"\"\"
    print(f"Loading Retrieval DBs: {args.retrieval_datasets}...")
    dataset_splits = {}

    for ds in args.retrieval_datasets:
        paths = []
        if ds == 'aircraft':
            root = "datasets/fgvc-aircraft-2013b/data/images_nobg" if use_nobg else "datasets/fgvc-aircraft-2013b/data/images"
            list_file = "datasets/fgvc-aircraft-2013b/data/images_train.txt"
            if os.path.exists(list_file):
                with open(list_file, 'r') as f:
                    paths = [os.path.join(root, line.strip() + ".jpg") for line in f if line.strip()]
        elif ds == 'cub':
            root = "datasets/CUB_200_2011/images"
            split_file = "datasets/CUB_200_2011/train_test_split.txt"
            images_file = "datasets/CUB_200_2011/images.txt"
            if os.path.exists(split_file) and os.path.exists(images_file):
                train_ids = set()
                with open(split_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2 and parts[1] == '1':
                            train_ids.add(parts[0])
                with open(images_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2 and parts[0] in train_ids:
                            paths.append(os.path.join(root, parts[1]))
        elif ds == 'imagenet':
            root = "/home/tingyu/imageRAG/datasets/ILSVRC2012_train"
            list_file = "/home/tingyu/imageRAG/datasets/imagenet_train_list.txt"
            if os.path.exists(list_file):
                print(f"[System] Reading ImageNet paths directly, avoiding slow disk validation...")
                with open(list_file, 'r') as f:
                    # Skip os.path.exists validation for 1.2M files!
                    paths = [os.path.join(root, line.strip()) for line in f if line.strip()]

        dataset_splits[ds] = paths
        print(f"Loaded {len(paths)} intended paths for {ds}.")

    total = sum(len(p) for p in dataset_splits.values())
    print(f"Total structured retrieval images: {total}")
    return dataset_splits"""

text = re.sub(load_db_pattern, new_load_db.replace('\\', '\\\\'), text, flags=re.DOTALL)


# 3. FIX INIT RETRIEVER
init_retriever_pattern = r"""    # --- Initialize Retriever ---
    try:
        print\("\[Main\] Initializing ImageRetriever\.\.\."\)
        retriever = ImageRetriever\(.*?\)
        torch\.cuda\.empty_cache\(\)
    except Exception as e:
        print\(f"Warning: Retriever init failed: \{e\}"\)"""

new_init_retriever = """    # --- Initialize Retriever ---
    class MultiDatasetRetriever:
        def __init__(self, dataset_splits, method, base_device, k, use_hybrid, ext_model, ext_proc, adapter):
            self.retrievers = []
            self.k = k
            for ds_name, paths in dataset_splits.items():
                emb_path = f"datasets/embeddings/{ds_name}"
                print(f"[MultiRetriever] Initializing sub-retriever for {ds_name}...")
                self.retrievers.append(ImageRetriever(
                    image_paths=paths,
                    embeddings_path=emb_path,
                    method=method,
                    device=base_device,
                    k=k,
                    use_hybrid=use_hybrid,
                    external_model=ext_model,
                    external_processor=ext_proc,
                    adapter_path=adapter
                ))

        def search(self, queries):
            all_paths = []
            all_scores = []
            for r in self.retrievers:
                paths, scores = r.search(queries)
                all_paths.extend(paths)
                all_scores.extend(scores)

            if not all_paths: return [], []
            combined = list(zip(all_paths, all_scores))
            combined.sort(key=lambda x: x[1], reverse=True)
            top_k = combined[:self.k]
            return [x[0] for x in top_k], [x[1] for x in top_k]

    try:
        print("[Main] Initializing MultiDatasetRetriever...")
        retriever = MultiDatasetRetriever(
            dataset_splits=retrieval_db,
            method=args.retrieval_method,
            base_device=retrieval_device,
            k=args.var_k,
            use_hybrid=args.use_hybrid_retrieval,
            ext_model=GLOBAL_QWEN_MODEL if args.retrieval_method in ["Qwen3-VL", "Qwen2.5-VL"] else None,
            ext_proc=GLOBAL_QWEN_PROCESSOR if args.retrieval_method in ["Qwen3-VL", "Qwen2.5-VL"] else None,
            adapter=args.adapter_path,
        )
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Retriever init failed: {e}")"""

text = re.sub(init_retriever_pattern, new_init_retriever.replace('\\', '\\\\'), text, flags=re.DOTALL)


# 4. FIX PROXY ENV POPping
# Because os.environ pops HTTP_PROXY but it was done BEFORE imports, let's make sure it still happens at top.
proxy_pattern = r"""# \[Proxy Config\]
os\.environ\.pop\("HTTP_PROXY", None\)
os\.environ\.pop\("HTTPS_PROXY", None\)
os\.environ\.pop\("http_proxy", None\)
os\.environ\.pop\("https_proxy", None\)
os\.environ\["HF_ENDPOINT"\] = "https://hf-mirror.com\""""
# we can leave it untouched since it's just os.environ modifying at top

with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Refactor applied.")
