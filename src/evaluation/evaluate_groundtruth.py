
'''
GroundTruth Baseline Evaluation Script (EvalScope Version)
=========================================================
Evaluates test set images against their class prompts using EvalScope metrics.
This establishes a "perfect generation" baseline or upper bound for metrics like CLIPScore, VQAScore, etc.

python evaluate_groundtruth.py \
  --device_id 0 \
  --metrics all \
  --output_file groundtruth_metrics.txt
'''

import os
import sys
import argparse
import json
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
import multiprocessing
import gc
import torch

# --- Metric Definitions ---
METRIC_INFO = {
    "VQAScore": "Visual Question Answering (CLIP-FlanT5).",
    "CLIPScore": "Image-text alignment using CLIP.",
    "BLIPv2Score": "Image-text matching using BLIP-2.",
    "PickScore": "Preference model trained on Pick-a-Pic.",
    "HPSv2Score": "Human Preference Score v2.",
    "HPSv2.1Score": "Updated Human Preference Score v2.1.",
    "ImageReward": "Human preference reward model.",
    "MPS": "Multi-modal Preference Score.",
    "FGA_BLIP2Score": "Fine-Grained Alignment score."
}

def load_metric_evaluator(metric_name):
    """Factory function to load a single metric evaluator on demand."""
    try:
        # Import EvalScope (Handle structure variations)
        try:
            from evalscope.metrics import (
                CLIPScoreMetric, ImageRewardMetric, VQAScore, PickScore,
                BLIPv2Score, HPSv2Score, HPSv2_1Score, MPS, FGA_BLIP2Score
            )
        except ImportError:
            try:
                from evalscope.metrics.vl import (
                    CLIPScoreMetric, ImageRewardMetric, VQAScore, PickScore,
                    BLIPv2Score, HPSv2Score, HPSv2_1Score, MPS, FGA_BLIP2Score
                )
            except ImportError:
                from evalscope.metrics.metric import (
                    CLIPScore, ImageRewardScore, VQAScore, PickScore,
                    BLIPv2Score, HPSv2Score, HPSv2_1Score, MPS, FGA_BLIP2Score
                )
                CLIPScoreMetric = CLIPScore
                ImageRewardMetric = ImageRewardScore

        print(f"Loading {metric_name} model to GPU...")

        if metric_name == "CLIPScore": return CLIPScoreMetric()
        elif metric_name == "ImageReward": return ImageRewardMetric()
        elif metric_name == "PickScore": return PickScore()
        elif metric_name == "VQAScore": return VQAScore()
        elif metric_name == "BLIPv2Score": return BLIPv2Score()
        elif metric_name == "HPSv2Score": return HPSv2Score()
        elif metric_name == "HPSv2.1Score": return HPSv2_1Score()
        elif metric_name == "MPS": return MPS()
        elif metric_name == "FGA_BLIP2Score": return FGA_BLIP2Score()
        else:
            print(f"Unknown metric: {metric_name}")
            return None
    except Exception as e:
        print(f"Failed to load {metric_name}: {e}")
        return None

def run_single_gt_metric(metric_name, tasks_file, result_file, device_id):
    """
    Worker function to run a single metric on Ground Truth images.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    # Load Tasks
    with open(tasks_file, 'r') as f:
        tasks = json.load(f)

    # Check if already done
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                existing_res = json.load(f)
                if metric_name in existing_res:
                    print(f"[{metric_name}] Already computed. Skipping.")
                    return
        except: pass

    # Load Model
    evaluator = load_metric_evaluator(metric_name)
    if evaluator is None: return

    scores = []

    print(f"[{metric_name}] Evaluating {len(tasks)} samples...")
    for item in tqdm(tasks, desc=metric_name):
        try:
            image_path = item['image_path']
            prompt = item['prompt']

            if not os.path.exists(image_path): continue

            res = evaluator(image=image_path, text=prompt)

            # Unpack value
            val = 0.0
            if isinstance(res, dict): val = list(res.values())[0]
            elif isinstance(res, (list, tuple)): val = res[0]
            else: val = res

            val = float(val)
            scores.append(val)

        except Exception as e:
            # print(f"Error processing {image_path}: {e}")
            pass

    # Save Result
    mean_score = float(np.mean(scores)) if scores else 0.0
    print(f"[{metric_name}] Mean Score: {mean_score}")

    # Update result file safely
    try:
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        data[metric_name] = mean_score
        with open(result_file, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving results: {e}")

    del evaluator
    gc.collect()
    torch.cuda.empty_cache()


def main():
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Evaluate GroundTruth with EvalScope Metrics")
    parser.add_argument("--device_id", type=str, default="0")
    parser.add_argument("--classes_txt", type=str, default="datasets/fgvc-aircraft-2013b/data/variants.txt")
    parser.add_argument("--real_images_list", type=str, default="datasets/fgvc-aircraft-2013b/data/images_variant_test.txt")
    parser.add_argument("--real_images_root", type=str, default="datasets/fgvc-aircraft-2013b/data/images")
    parser.add_argument("--output_file", type=str, default="groundtruth_metrics.txt")
    parser.add_argument("--samples_per_class", type=int, default=5, help="Number of real images to sample per class")
    parser.add_argument("--metrics", type=str, default="all", help="Comma-sep list or 'all'")

    args = parser.parse_args()

    # 1. Prepare Tasks
    print("Preparing GroundTruth tasks...")

    class_names = []
    if os.path.exists(args.classes_txt):
        with open(args.classes_txt, 'r') as f:
            class_names = [l.strip() for l in f.readlines() if l.strip()]
    else:
        print(f"Error: {args.classes_txt} not found.")
        return

    class_to_images = defaultdict(list)
    if os.path.exists(args.real_images_list):
        with open(args.real_images_list, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) >= 2:
                    img_id = parts[0]
                    cls_name = parts[1]
                    if cls_name in class_names:
                        full_path = os.path.join(args.real_images_root, f"{img_id}.jpg")
                        if os.path.exists(full_path):
                            class_to_images[cls_name].append(full_path)
    else:
        print(f"Error: {args.real_images_list} not found.")
        return

    task_list = []
    print(f"Found {len(class_to_images)} classes with images.")

    for cls in class_names:
        imgs = class_to_images.get(cls, [])
        if not imgs: continue

        # Determine number of samples
        n_samples = args.samples_per_class
        if n_samples <= 0:  # If 0 or negative, use all
            selected = imgs
        else:
            selected = random.sample(imgs, min(len(imgs), n_samples))

        for img_path in selected:
            task_list.append({
                "prompt": f"a photo of a {cls}",
                "image_path": img_path,
                "class": cls
            })

    print(f"Total GroundTruth Samples to Evaluate: {len(task_list)}")
    if not task_list:
        print("No tasks found. Exiting.")
        return

    # Save tasks to temp file
    temp_task_file = "temp_gt_tasks.json"
    with open(temp_task_file, 'w') as f:
        json.dump(task_list, f)

    # Temp result file
    temp_result_file = "temp_gt_results.json"
    if not os.path.exists(temp_result_file):
        with open(temp_result_file, 'w') as f:
            json.dump({}, f)

    # 2. Determine Metrics
    metric_names = [m.strip() for m in args.metrics.split(",")]
    if "all" in metric_names:
        metric_names = [
            "CLIPScore", "ImageReward", "PickScore",
            "VQAScore", "BLIPv2Score", "HPSv2Score",
            "HPSv2.1Score", "MPS", "FGA_BLIP2Score"
        ]

    # 3. Run Metrics
    processed_count = 0
    for m in metric_names:
        print(f"\n[Manager] Starting Evaluation for {m}...")
        p = multiprocessing.Process(
            target=run_single_gt_metric,
            args=(m, temp_task_file, temp_result_file, args.device_id)
        )
        p.start()
        p.join()

        if p.exitcode != 0:
            print(f"[Manager] Metric {m} failed/crashed.")
        else:
            processed_count += 1

    # 4. Finalize
    print(f"\nEvaluation Finished. Processed {processed_count} metrics.")
    print(f"Writing final report to {args.output_file}")

    final_output = {}
    if os.path.exists(temp_result_file):
        with open(temp_result_file, 'r') as f:
            final_output = json.load(f)

    with open(args.output_file, 'w') as f:
        for k, v in final_output.items():
            f.write(f"{k}: {v:.4f}\n")
            print(f"  {k}: {v:.4f}")

    # Cleanup
    if os.path.exists(temp_task_file): os.remove(temp_task_file)
    if os.path.exists(temp_result_file): os.remove(temp_result_file)

if __name__ == "__main__":
    main()
