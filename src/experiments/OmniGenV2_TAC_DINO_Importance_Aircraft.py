'''
OmniGenV2_TAC_DINO_Importance_Aircraft.py
==========================================
Importance-Based 5-Step Pipeline (重要度划分全新流水线)

Architecture:
  Step 1: Input Interpreter (非对称解析)
    - Forced subject/context decomposition with importance weighting
    - high_importance: pure entity name (for retrieval)
    - low_importance: environment/style/composition (for generation context)

  Step 2: Dual-Stage Retrieval (护城河过滤)
    - Stage A: LongCLIP recall with pure entity only → Top-K candidates
    - Stage B: VAR reranking via Qwen3-VL for cleanest subject ("极品证件照")
    - Output: best reference image (highest subject fidelity)

  Step 3: Initial Generation (结构先验注入 / DINO-injected)
    - DINOv3 2D patch feature extraction from reference image
    - Inject structural prior into OmniGen2 DiT via Late Fusion
    - Full prompt (high + low importance) used for text guidance

  Step 4: TAC Spatial Diagnosis (空间级诊断)
    - Taxonomy-Aware Critic with hard knowledge specs
    - Score >= 6.0 (Tier A) → pass; otherwise → Step 5
    - Spatial error analysis (Global vs Local)

  Step 5: Reflexive Re-generation (带外挂的重试)
    - Negative prompt injection from TAC diagnosis
    - DINO adapter weight λ escalation (increase structural constraint)
    - Retry with refined prompt + stronger DINO guidance

Configuration:
  - Generator: OmniGen V2 (DiT Late Fusion) + DINOv3 Structure Prior
  - Critic: Taxonomy-Aware Critic (TAC) with spatial diagnosis
  - Retrieval: Importance-Aware Dual-Stage (LongCLIP + VAR)
  - Dataset: FGVC-Aircraft

Usage:
  python OmniGenV2_TAC_DINO_Importance_Aircraft.py \
      --device_id 0 \
      --vlm_device_id 1 \
      --task_index 0 \
      --total_chunks 1 \
      --omnigen2_path ./OmniGen2 \
      --retrieval_method LongCLIP \
      --use_local_model_weight

  Notes:
    --enable_offload and --enable_taylorseer are ON by default.
    --enable_teacache and --qwen_4bit are OFF by default.
    Use --disable_offload / --disable_taylorseer to turn them off.
'''

# ==============================================================================
# Imports
# ==============================================================================
from datetime import datetime
import argparse
import sys
import os

# Wait to import torch until CUDA is set

# ==============================================================================
# Argument Parsing
# ==============================================================================
parser = argparse.ArgumentParser(description="OmniGenV2 + TAC + DINO Importance Pipeline (Aircraft)")

# Core Config
parser.add_argument("--device_id", type=str, required=True, help="Main device ID (e.g. '0' or '0,1')")
parser.add_argument("--vlm_device_id", type=str, default=None, help="Device ID for VLM (if different)")
parser.add_argument("--task_index", type=int, default=0)
parser.add_argument("--total_chunks", type=int, default=1)

# Paths
parser.add_argument("--omnigen2_path", type=str, default="./OmniGen2")
parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2")
parser.add_argument("--transformer_lora_path", type=str, default=None)
parser.add_argument("--openai_api_key", type=str, required=False)
parser.add_argument("--llm_model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")

# Local Weights Config
parser.add_argument("--use_local_model_weight", action="store_true")
parser.add_argument("--local_model_weight_path", type=str, default="/home/tingyu/imageRAG/Qwen3-VL-4B-Instruct")
parser.add_argument("--enable_offload", action="store_true", default=True,
                    help="Enable CPU offloading to reduce VRAM (default: ON)")
parser.add_argument("--disable_offload", action="store_false", dest="enable_offload",
                    help="Disable CPU offloading")
parser.add_argument("--enable_teacache", action="store_true", default=False,
                    help="Enable TeaCache for faster inference (default: OFF, mutually exclusive with TaylorSeer)")
parser.add_argument("--teacache_thresh", type=float, default=0.4)
parser.add_argument("--enable_taylorseer", action="store_true", default=True,
                    help="Enable TaylorSeer for ~2x faster inference (default: ON)")
parser.add_argument("--disable_taylorseer", action="store_false", dest="enable_taylorseer",
                    help="Disable TaylorSeer")
parser.add_argument("--qwen_4bit", action="store_true", default=False,
                    help="Enable 4-bit quantization for Qwen3-VL (default: OFF)")
parser.add_argument("--retrieval_cpu", action="store_true")

# Generation Params
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_retries", type=int, default=3, help="Max reflexive re-generation attempts (Step 5)")
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--image_guidance_scale", type=float, default=1.6)
parser.add_argument("--text_guidance_scale", type=float, default=2.5)
parser.add_argument("--use_negative_prompt", action="store_true", default=False,
                    help="Enable negative prompt injection (default: False)")
parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, text, watermark, lowres, ugly, deformed")

# Retrieval
parser.add_argument("--embeddings_path", type=str, default="datasets/embeddings/aircraft")
parser.add_argument("--retrieval_method", type=str, default="LongCLIP",
                    choices=["CLIP", "LongCLIP", "SigLIP", "SigLIP2", "Qwen2.5-VL", "Qwen3-VL"])
parser.add_argument("--use_hybrid_retrieval", action="store_true")
parser.add_argument("--adapter_path", type=str, default=None)
parser.add_argument("--retrieval_datasets", nargs='+', default=['aircraft'],
                    choices=['aircraft', 'cub', 'imagenet'])
parser.add_argument("--var_k", type=int, default=10, help="Number of candidates for VAR reranking")
parser.add_argument("--use_nobg", action="store_true")

# DINO Config (New)
parser.add_argument("--dino_model_path", type=str, default="dinov3/",
                    help="Path to DINOv3 source directory")
parser.add_argument("--dino_weights_path", type=str, default="dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
                    help="Path to DINOv3 pretrained weights")
parser.add_argument("--dino_lambda_init", type=float, default=0.3,
                    help="Initial DINO structural prior injection weight (λ)")
parser.add_argument("--dino_lambda_step", type=float, default=0.15,
                    help="DINO λ escalation per retry step")
parser.add_argument("--dino_lambda_max", type=float, default=0.8,
                    help="Maximum DINO λ clamp value")

# Importance Threshold
parser.add_argument("--tac_pass_threshold", type=float, default=6.0,
                    help="TAC score threshold for accepting generation (Tier A)")
parser.add_argument("--tac_early_stop_threshold", type=float, default=8.0,
                    help="TAC score threshold for early stopping (excellent)")

# Timestep Decoupling & SAM2
parser.add_argument("--use_sam2_matting", action="store_true", default=False,
                    help="Enable SAM2 background matting for extreme pureness")
parser.add_argument("--decouple_threshold", type=float, default=0.25,
                    help="Timestep decay threshold for DINO/ref latents (0.0=disabled)")

args = parser.parse_args()

# ==============================================================================
# Device Setup
# ==============================================================================
if args.vlm_device_id:
    if args.device_id == args.vlm_device_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
        omnigen_device = "cuda:0"
        vlm_device_map = {"": "cuda:0"}
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device_id},{args.vlm_device_id}"
        omnigen_device = "cuda:0"
        vlm_device_map = {"": "cuda:1"}
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    omnigen_device = "cuda:0"
    vlm_device_map = "auto"

print(f"[Config] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

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


# ==============================================================================
# Project Imports
# ==============================================================================
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.critical.taxonomy_aware_critic import (
    taxonomy_aware_diagnosis,
    generate_knowledge_specs,
    input_interpreter,
    message_gpt,
    encode_image,
)
from src.retrieval.memory_guided_retrieval import ImageRetriever
from src.utils.rag_utils import (
    LocalQwen3VLWrapper,
    UsageTrackingClient,
    ResourceMonitor,
    RUN_STATS,
    seed_everything,
)


# ==============================================================================
# DINOv3 Feature Extractor
# ==============================================================================
class DINOv3FeatureExtractor:
    """
    Extracts 2D patch-level features from DINOv3 ViT-B/16.
    Used as structural prior for DiT injection.
    """
    def __init__(self, model_path, weights_path, device="cuda"):
        self.device = device
        self.model = None
        self.transform = None
        self._load_model(model_path, weights_path)

    def _load_model(self, model_path, weights_path):
        """Load DINOv3 model using torch.hub local source (same as evaluate_all_recursive.py)."""
        try:
            import types
            print(f"[DINO] Loading DINOv3 from {model_path}...")

            # Mock dinov3.data modules to avoid dataset dependency errors
            sys.path.append(model_path)
            mock_data = types.ModuleType("dinov3.data")
            mock_datasets = types.ModuleType("dinov3.data.datasets")
            class DatasetWithEnumeratedTargets: pass
            class SamplerType: pass
            class ImageDataAugmentation: pass
            def make_data_loader(*a, **kw): pass
            mock_data.DatasetWithEnumeratedTargets = DatasetWithEnumeratedTargets
            mock_data.SamplerType = SamplerType
            mock_data.ImageDataAugmentation = ImageDataAugmentation
            mock_data.make_data_loader = make_data_loader
            mock_data.datasets = mock_datasets
            sys.modules["dinov3.data"] = mock_data
            sys.modules["dinov3.data.datasets"] = mock_datasets

            self.model = torch.hub.load(model_path, "dinov3_vitb16", source="local", pretrained=False)

            if os.path.exists(weights_path):
                print(f"[DINO] Loading weights from {weights_path}...")
                ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
                state_dict = ckpt.get('model', ckpt.get('teacher', ckpt))
                new_state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v
                                  for k, v in state_dict.items()}
                self.model.load_state_dict(new_state_dict, strict=False)

            self.model.eval().to(self.device)

            # Standard ImageNet normalization for DINOv3
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            print("[DINO] DINOv3 model loaded successfully.")
        except Exception as e:
            print(f"[DINO] Error loading DINOv3: {e}")
            self.model = None

    @torch.no_grad()
    def extract_patch_features(self, image_path):
        """
        Extract 2D patch-level features from a reference image.

        Returns:
            patch_features: Tensor [1, N_patches, D] where N_patches = (H/patch_size)*(W/patch_size)
            cls_token: Tensor [1, D] global CLS token
        """
        if self.model is None:
            return None, None

        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # Get intermediate features (patch tokens)
            # DINOv3 forward returns CLS token by default
            # Use get_intermediate_layers for patch tokens
            if hasattr(self.model, 'get_intermediate_layers'):
                # Returns list of [B, N_patches, D]
                features = self.model.get_intermediate_layers(img_tensor, n=1)
                patch_features = features[0]  # Last layer: [1, N_patches, D]
            else:
                # Fallback: standard forward
                output = self.model(img_tensor)
                patch_features = output.unsqueeze(1) if output.dim() == 2 else output

            # CLS token (global descriptor)
            cls_token = self.model(img_tensor)
            if cls_token.dim() == 2:
                cls_token = cls_token  # [1, D]
            else:
                cls_token = cls_token[:, 0]

            return patch_features, cls_token

        except Exception as e:
            print(f"[DINO] Feature extraction error: {e}")
            return None, None

    @torch.no_grad()
    def compute_similarity(self, image_path_a, image_path_b):
        """Compute DINOv3 cosine similarity between two images."""
        _, cls_a = self.extract_patch_features(image_path_a)
        _, cls_b = self.extract_patch_features(image_path_b)

        if cls_a is None or cls_b is None:
            return 0.0

        sim = F.cosine_similarity(cls_a, cls_b, dim=-1)
        return sim.item()

    def to(self, device):
        """Move model to specified device."""
        if self.model is not None:
            self.model.to(device)
            self.device = device if isinstance(device, str) else str(device)
        return self


# ==============================================================================
# SAM2 Physical Matting (For DINO pureness only)
# ==============================================================================
def apply_sam2_matting(image_path, output_path, device="cuda"):
    import os
    import sys
    import numpy as np
    from PIL import Image

    orig_dir = os.getcwd()
    # Cache original sys.path to restore later
    old_sys_path = sys.path.copy()

    try:
        # Absolutize paths before changing directories
        if not os.path.isabs(image_path):
            image_path = os.path.join(orig_dir, image_path)
        if not os.path.isabs(output_path):
            output_path = os.path.join(orig_dir, output_path)

        # Remove parent directories from sys.path to prevent namespace shadowing (SAM2 strict requirement)
        sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(orig_dir) and p not in ("", ".")]

        repo_dir = os.path.join(orig_dir, "sam2")
        sys.path.insert(0, repo_dir)
        os.chdir(repo_dir)

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # Using SAM 2.1 Base Plus as requested
        sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        predictor.set_image(image_np)

        H, W, _ = image_np.shape
        input_point = np.array([[W//2, H//2]])
        input_label = np.array([1])

        # [Fix] Enable multimask to solve local-global ambiguity
        masks, scores, logits = predictor.predict(
            point_coords=input_point, point_labels=input_label, multimask_output=True,
        )

        # Calculate areas (number of True/1 pixels) and pick the largest one (whole aircraft)
        areas = [mask.sum() for mask in masks]
        largest_idx = np.argmax(areas)
        best_mask = masks[largest_idx] > 0.0

        image_np[~best_mask] = [255, 255, 255] # 背景涂白 (隔离背景毒害)

        Image.fromarray(image_np).save(output_path)
        del sam2_model, predictor
        import torch
        torch.cuda.empty_cache()
        return output_path
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"[SAM2] Matting failed:\n{error_msg}")
        return None
    finally:
        os.chdir(orig_dir)
        sys.path = old_sys_path


# ==============================================================================
# Step 1: Importance-Aware Input Interpreter (非对称解析)
# ==============================================================================
def importance_aware_input_interpreter(prompt, client, model="gpt-4o", domain="aircraft"):
    """
    Stage 1: Decompose user prompt into importance-weighted components.

    Returns:
        dict with keys:
        - high_importance: pure entity name (for retrieval)
        - low_importance: environment/style/composition details (for generation)
        - retrieval_query: optimized query for image retrieval (entity-only)
        - generation_prompt: full detailed prompt for image generation
        - importance_weights: dict of relative importance scores
    """
    domain_context = {
        "aircraft": "aircraft identification and aviation photography",
        "birds": "ornithology and bird photography",
        "cub": "ornithology and bird photography",
    }.get(domain, "fine-grained visual recognition")

    msg = f"""
    You are an expert Input Interpreter Agent specialized in {domain_context}.

    **Task:** Decompose the following prompt into importance-weighted components for a
    text-to-image generation pipeline with Retrieval-Augmented Generation (RAG).

    **User Prompt:** "{prompt}"

    **Requirements:**

    1. **High Importance (Entity / Subject)** — This is the CORE visual identity.
       - Extract the EXACT subject entity name (e.g., "Boeing 707-320", "Airbus A380").
       - This will be used ALONE for retrieval search. It MUST be a clean, specific name.
       - NO style words, NO environment words. Just the pure entity.

    2. **Low Importance (Context / Style)** — This is for creative enrichment ONLY.
       - Background, lighting, composition, artistic style, camera angle.
       - These NEVER affect retrieval, only affect the final generation prompt.

    3. **Importance Weights** — Assign relative importance (0.0-1.0):
       - entity_identity: How important is getting the exact entity right (usually 0.8-1.0)
       - structural_detail: How important are fine structural details (e.g., engine count)
       - environment: How important is the background/scene
       - artistic_style: How important is the visual style

    4. **Retrieval Query** — A clean, entity-focused query string.
       Format: "a photo of a [EXACT ENTITY NAME]" — nothing else.

    5. **Generation Prompt** — A full, detailed prompt combining ALL elements for T2I model.
       Include high-quality photorealistic details and all creative fills.

    **Output JSON Format:**
    {{
        "high_importance": {{
            "entity": "exact entity name",
            "key_features": ["feature1", "feature2"]
        }},
        "low_importance": {{
            "background": "...",
            "composition": "...",
            "lighting": "...",
            "visual_style": "..."
        }},
        "importance_weights": {{
            "entity_identity": 0.9,
            "structural_detail": 0.8,
            "environment": 0.3,
            "artistic_style": 0.2
        }},
        "retrieval_query": "a photo of a [entity]",
        "generation_prompt": "full detailed prompt...",
        "ambiguous_elements": ["any ambiguities detected"]
    }}
    """

    ans_text = message_gpt(msg, client, [], model=model)

    try:
        import ast as _ast
        import re as _re

        def _extract_outer_json_block(text):
            if not text:
                return text
            start = text.find("{")
            if start < 0:
                return text
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:idx + 1]
            return text[start:]

        def _sanitize_json_like(text):
            fixed = text.strip()
            fixed = fixed.replace("\u201c", '"').replace("\u201d", '"')
            fixed = fixed.replace("\u2018", "'").replace("\u2019", "'")
            fixed = fixed.replace("：", ":").replace("，", ",")
            fixed = fixed.replace("\\n", " ")
            fixed = _re.sub(r"[\x00-\x1F\x7F-\x9F]", "", fixed)
            fixed = _re.sub(r",\s*([}\]])", r"\1", fixed)
            return fixed

        def _parse_json_loose(text):
            candidates = []
            if text:
                candidates.append(text)
                candidates.append(_sanitize_json_like(text))

            for candidate in candidates:
                try:
                    return json.loads(candidate)
                except Exception:
                    pass

            for candidate in candidates:
                try:
                    py_candidate = candidate
                    py_candidate = _re.sub(r"\btrue\b", "True", py_candidate, flags=_re.IGNORECASE)
                    py_candidate = _re.sub(r"\bfalse\b", "False", py_candidate, flags=_re.IGNORECASE)
                    py_candidate = _re.sub(r"\bnull\b", "None", py_candidate, flags=_re.IGNORECASE)
                    parsed = _ast.literal_eval(py_candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            return None

        def _extract_str(field_name, text):
            pattern = rf'["\']{field_name}["\']\s*[:=]\s*["\']([^"\']+)["\']'
            match = _re.search(pattern, text, _re.IGNORECASE | _re.DOTALL)
            return match.group(1).strip() if match else None

        def _extract_float(field_name, text):
            pattern = rf'["\']{field_name}["\']\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)'
            match = _re.search(pattern, text, _re.IGNORECASE)
            return float(match.group(1)) if match else None

        def _extract_list(field_name, text):
            pattern = rf'["\']{field_name}["\']\s*[:=]\s*\[(.*?)\]'
            match = _re.search(pattern, text, _re.IGNORECASE | _re.DOTALL)
            if not match:
                return None
            raw = match.group(1)
            parts = [p.strip().strip('"\' ') for p in raw.split(",") if p.strip()]
            return [p for p in parts if p]

        clean_text = ans_text.strip()
        fenced_match = _re.search(r"```(?:json)?\s*(.*?)\s*```", ans_text, _re.DOTALL | _re.IGNORECASE)
        if fenced_match:
            clean_text = fenced_match.group(1).strip()

        clean_text = _extract_outer_json_block(clean_text)
        result = _parse_json_loose(clean_text)

        if result is None:
            print("[InputInterpreter Info] Attempting aggressive field-level recovery...")
            result = {}
            entity = _extract_str("entity", clean_text)
            key_features = _extract_list("key_features", clean_text) or []

            low_importance = {
                "background": _extract_str("background", clean_text) or "clear sky, runway",
                "composition": _extract_str("composition", clean_text) or "side view",
                "lighting": _extract_str("lighting", clean_text) or "natural daylight",
                "visual_style": _extract_str("visual_style", clean_text) or "photorealistic",
            }

            importance_weights = {
                "entity_identity": _extract_float("entity_identity", clean_text) or 0.9,
                "structural_detail": _extract_float("structural_detail", clean_text) or 0.8,
                "environment": _extract_float("environment", clean_text) or 0.3,
                "artistic_style": _extract_float("artistic_style", clean_text) or 0.2,
            }

            retrieval_query = _extract_str("retrieval_query", clean_text)
            generation_prompt = _extract_str("generation_prompt", clean_text)
            ambiguous_elements = _extract_list("ambiguous_elements", clean_text) or []

            entity_clean = (entity or prompt).replace("a photo of a ", "").replace("a photo of ", "").strip()

            result["high_importance"] = {
                "entity": entity_clean,
                "key_features": key_features,
            }
            result["low_importance"] = low_importance
            result["importance_weights"] = importance_weights
            result["retrieval_query"] = retrieval_query or f"a photo of a {entity_clean}"
            result["generation_prompt"] = generation_prompt or prompt
            result["ambiguous_elements"] = ambiguous_elements

        if not isinstance(result.get("high_importance"), dict):
            result["high_importance"] = {}
        if not isinstance(result.get("low_importance"), dict):
            result["low_importance"] = {}
        if not isinstance(result.get("importance_weights"), dict):
            result["importance_weights"] = {}

        entity = result.get("high_importance", {}).get("entity", "")
        if not entity:
            entity = prompt.replace("a photo of a ", "").replace("a photo of ", "").strip()
            result["high_importance"]["entity"] = entity

        if "key_features" not in result["high_importance"] or not isinstance(result["high_importance"].get("key_features"), list):
            result["high_importance"]["key_features"] = []

        lo_defaults = {
            "background": "clear sky, runway",
            "composition": "side view",
            "lighting": "natural daylight",
            "visual_style": "photorealistic",
        }
        for key, default_val in lo_defaults.items():
            if not result["low_importance"].get(key):
                result["low_importance"][key] = default_val

        iw_defaults = {
            "entity_identity": 0.9,
            "structural_detail": 0.8,
            "environment": 0.3,
            "artistic_style": 0.2,
        }
        for key, default_val in iw_defaults.items():
            val = result["importance_weights"].get(key)
            if not isinstance(val, (int, float)):
                result["importance_weights"][key] = default_val

        if not result.get("retrieval_query"):
            result["retrieval_query"] = f"a photo of a {entity}"
        if not result.get("generation_prompt"):
            result["generation_prompt"] = prompt
        if "ambiguous_elements" not in result or not isinstance(result.get("ambiguous_elements"), list):
            result["ambiguous_elements"] = []

        return result

    except Exception as e:
        preview = (ans_text or "")[:300].replace("\n", "\\n")
        print(f"[InputInterpreter] Parse error: {e}")
        print(f"[InputInterpreter Debug] Raw response preview: {preview}")
        # Fallback: simple heuristic decomposition
        entity = prompt.replace("a photo of a ", "").replace("a photo of ", "").strip()
        return {
            "high_importance": {
                "entity": entity,
                "key_features": []
            },
            "low_importance": {
                "background": "clear sky, runway",
                "composition": "side view",
                "lighting": "natural daylight",
                "visual_style": "photorealistic"
            },
            "importance_weights": {
                "entity_identity": 0.9,
                "structural_detail": 0.8,
                "environment": 0.3,
                "artistic_style": 0.2
            },
            "retrieval_query": f"a photo of a {entity}",
            "generation_prompt": prompt,
            "ambiguous_elements": []
        }


# ==============================================================================
# Step 2: Dual-Stage Retrieval (护城河过滤)
# ==============================================================================
def dual_stage_retrieval(retrieval_query, retriever, client, model_name,
                         var_k=10, f_log=None):
    """
    Stage 2: Importance-Aware Dual-Stage Retrieval.

    Stage A: Use pure entity query with base retriever (LongCLIP/CLIP) → Top-K
    Stage B: VLM-As-Reranker (VAR) verifies each candidate for subject fidelity
             → Select the cleanest "证件照" (ID-photo quality subject match)

    Args:
        retrieval_query: Pure entity query (from Step 1 high_importance)
        retriever: ImageRetriever instance
        client: VLM client for VAR reranking
        model_name: VLM model name
        var_k: Number of candidates to retrieve for VAR stage

    Returns:
        best_ref: Path to best reference image
        best_ref_score: Similarity score
        var_details: Dict with verification details
    """
    log = lambda msg: (f_log.write(msg + "\n") if f_log else print(msg))

    log(f"  [Step2-A] Base Retrieval: query='{retrieval_query}' (K={var_k})")

    try:
        retrieved_lists, retrieved_scores = retriever.search([retrieval_query])

        if not retrieved_lists or not retrieved_lists[0]:
            log("  [Step2-A] WARNING: No retrieval results!")
            return None, 0.0, {"stage_a_count": 0, "stage_b_passed": 0}

        candidates = retrieved_lists[0][:var_k]
        candidate_scores = retrieved_scores[0][:var_k]

        log(f"  [Step2-A] Retrieved {len(candidates)} candidates.")

    except Exception as e:
        log(f"  [Step2-A] Retrieval error: {e}")
        return None, 0.0, {"stage_a_count": 0, "stage_b_passed": 0, "error": str(e)}

    # Stage B: VAR Reranking (VLM verification for subject fidelity)
    log(f"  [Step2-B] VAR Reranking with VLM...")

    valid_refs = []
    best_ref_score = 0.0
    var_details = {
        "stage_a_count": len(candidates),
        "stage_b_passed": 0,
        "stage_b_results": []
    }

    entity_name = retrieval_query.replace("a photo of a ", "").replace("a photo of ", "").strip()

    for idx, (c_path, c_score) in enumerate(zip(candidates, candidate_scores)):
        try:
            # VAR verification prompt: focus on subject identity, not style
            check_prompt = (
                f"Look at this image carefully. Does it show a '{entity_name}'? "
                f"Focus ONLY on the subject identity — ignore background, lighting, or artistic quality. "
                f"Is the subject clearly and correctly a '{entity_name}'? "
                f"Answer with 'Yes' or 'No', then briefly explain why."
            )

            # Encode image for VLM
            img_base64 = encode_image(c_path)
            if img_base64 is None:
                log(f"    [#{idx}] SKIP - Cannot encode image")
                continue

            resp = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                        {"type": "text", "text": check_prompt}
                    ]
                }],
                temperature=0.01
            )
            ans = resp.choices[0].message.content

            passed = any(kw in ans.lower() for kw in ["yes", "是", "correct", "matches"])
            var_details["stage_b_results"].append({
                "path": os.path.basename(c_path),
                "base_score": c_score,
                "vlm_answer": ans[:100],
                "passed": passed
            })

            if passed:
                var_details["stage_b_passed"] += 1
                valid_refs.append(c_path)
                if len(valid_refs) == 1:
                    # Take the first passing candidate (highest base retrieval score)
                    best_ref_score = c_score
                    log(f"    [#{idx}] PASS ✓ → Selected as best ref ({os.path.basename(c_path)})")
                else:
                    log(f"    [#{idx}] PASS ✓ (backup: {os.path.basename(c_path)})")
            else:
                log(f"    [#{idx}] FAIL ✗ ({ans[:80]})")

        except Exception as e:
            log(f"    [#{idx}] ERROR: {e}")
            var_details["stage_b_results"].append({
                "path": os.path.basename(c_path),
                "error": str(e)
            })

    if not valid_refs:
        log(f"  [Step2-B] DANGER: No candidate passed VAR. Abandoning structural retrieval!")
        return [], 0.0, var_details

    log(f"  [Step2] Final highest ref: {os.path.basename(valid_refs[0])} "
        f"(score={best_ref_score:.4f}, total valid backups: {len(valid_refs)-1})")

    return valid_refs, best_ref_score, var_details


# ==============================================================================
# Step 3: DINO-Injected Generation (结构先验注入)
# ==============================================================================
def dino_injected_generation(pipe, generation_prompt, ref_image_path,
                              dino_extractor, dino_lambda,
                              seed, omnigen_device,
                              output_path, f_log=None,
                              height=1024, width=1024,
                              img_guidance_scale=1.6,
                              text_guidance_scale=2.5,
                              negative_prompt="",
                              decouple_threshold=0.25,
                              dino_ref_path=None):
    """
    Stage 3: Generate image with DINOv3 structural prior injection.

    The DINOv3 patch features serve as a structural backbone:
    - Extract 2D patch features from reference image
    - Use them to modulate the image guidance scale
    - The structural similarity is injected via reference image + scaled guidance

    Args:
        pipe: OmniGen2 pipeline
        generation_prompt: Full generation prompt (from Step 1)
        ref_image_path: Best reference image (from Step 2)
        dino_extractor: DINOv3FeatureExtractor instance
        dino_lambda: Current DINO injection weight (λ)
        seed: Random seed
        omnigen_device: Target device
        output_path: Where to save generated image
        f_log: Log file handle
    """
    log = lambda msg: (f_log.write(msg + "\n") if f_log else print(msg))

    # Extract DINO features from reference for structural guidance
    dino_sim_score = 0.0
    effective_img_guidance = img_guidance_scale

    target_dino_path = dino_ref_path if dino_ref_path else ref_image_path
    if dino_extractor is not None and target_dino_path is not None:
        try:
            patch_features, cls_token = dino_extractor.extract_patch_features(target_dino_path)
            if cls_token is not None:
                # Modulate image guidance scale based on DINO structural confidence
                # Higher λ → stronger structural constraint from reference
                # image_guidance = base_guidance * (1 + λ * structural_confidence)
                structural_confidence = cls_token.norm(dim=-1).mean().item()
                # Normalize structural_confidence to [0, 1] range (typical range for ViT-B is ~15-25)
                structural_confidence = min(structural_confidence / 20.0, 1.0)

                effective_img_guidance = img_guidance_scale * (1.0 + dino_lambda * structural_confidence)
                effective_img_guidance = min(effective_img_guidance, 3.0)  # Clamp to prevent artifacts

                log(f"  [Step3-DINO] λ={dino_lambda:.2f}, structural_conf={structural_confidence:.3f}, "
                    f"effective_img_guidance={effective_img_guidance:.3f}")
            else:
                log("  [Step3-DINO] No DINO features extracted, using base guidance.")
        except Exception as e:
            log(f"  [Step3-DINO] Feature extraction error: {e}. Using base guidance.")

    # Prepare generation prompt with reference image tag
    if ref_image_path:
        gen_prompt = f"{generation_prompt}. Use reference image <|image_1|> for structural guidance."
    else:
        gen_prompt = generation_prompt

    log(f"  [Step3] Gen prompt: {gen_prompt[:200]}...")
    log(f"  [Step3] img_guidance={effective_img_guidance:.2f}, text_guidance={text_guidance_scale:.2f}")

    # Move helpers to CPU to free VRAM
    move_helpers_to_cpu()

    # Prepare input images
    input_images = []
    if ref_image_path and os.path.exists(ref_image_path):
        ref_img = Image.open(ref_image_path).convert("RGB")
        input_images.append(ref_img)

    gen = torch.Generator("cuda").manual_seed(seed)

    try:
        result = pipe(
            prompt=gen_prompt,
            input_images=input_images if input_images else None,
            height=height,
            width=width,
            text_guidance_scale=text_guidance_scale,
            image_guidance_scale=effective_img_guidance,
            num_inference_steps=50,
            generator=gen,
            negative_prompt=negative_prompt if args.use_negative_prompt else None,
            decouple_threshold=decouple_threshold,
        )
        result.images[0].save(output_path)
        log(f"  [Step3] Image saved to {output_path}")

    except Exception as e:
        log(f"  [Step3] Generation error: {e}")
        # Fallback: generate without reference
        try:
            gen = torch.Generator("cuda").manual_seed(seed)
            result = pipe(
                prompt=generation_prompt,
                input_images=None,
                height=height, width=width,
                text_guidance_scale=text_guidance_scale,
                image_guidance_scale=1.0,
                num_inference_steps=50,
                generator=gen,
                negative_prompt=negative_prompt if args.use_negative_prompt else None,
                decouple_threshold=decouple_threshold,
            )
            result.images[0].save(output_path)
            log(f"  [Step3] Fallback generation saved to {output_path}")
        except Exception as e2:
            log(f"  [Step3] Fallback also failed: {e2}")

    return effective_img_guidance


# ==============================================================================
# Step 4: TAC Spatial Diagnosis (空间级诊断)
# ==============================================================================
def tac_spatial_diagnosis(prompt, image_path, client, model_name,
                          reference_specs=None, domain="aircraft",
                          dino_extractor=None, ref_image_path=None,
                          f_log=None):
    """
    Stage 4: Taxonomy-Aware Critic with spatial-level error diagnosis.

    Extends standard TAC with:
    - DINOv3 structural similarity score (quantitative)
    - Spatial error classification (Global vs Local)
    - Remediation strategy with importance-aware feedback

    Returns:
        diagnosis: TAC diagnosis dict (with additional DINO metrics)
    """
    log = lambda msg: (f_log.write(msg + "\n") if f_log else print(msg))

    # Move helpers back to GPU for critic
    move_helpers_to_gpu(retrieval_device)

    # Step 4a: Standard TAC Diagnosis
    diagnosis = taxonomy_aware_diagnosis(
        prompt, [image_path], client, model_name,
        reference_specs=reference_specs, domain=domain
    )

    score = diagnosis.get("final_score", 0)
    taxonomy_status = diagnosis.get("taxonomy_check", "unknown")
    error_type = diagnosis.get("error_analysis", {}).get("type", "Unknown")

    log(f"  [Step4-TAC] Score={score}, Taxonomy={taxonomy_status}, ErrorType={error_type}")
    log(f"  [Step4-TAC] Critique: {diagnosis.get('critique', 'N/A')[:200]}")

    # Step 4b: DINOv3 Structural Similarity (quantitative metric)
    dino_score = 0.0
    if dino_extractor is not None and ref_image_path is not None:
        try:
            dino_score = dino_extractor.compute_similarity(image_path, ref_image_path)
            diagnosis["dino_similarity"] = dino_score
            log(f"  [Step4-DINO] Structural similarity: {dino_score:.4f}")
        except Exception as e:
            log(f"  [Step4-DINO] Similarity computation error: {e}")

    # Step 4c: Importance-Aware Remediation
    # Classify needed fix strategy based on error type + importance weights
    if score < 4.0:
        diagnosis["fix_strategy"] = "GLOBAL_REWRITE"
        diagnosis["fix_severity"] = "critical"
        log("  [Step4] Fix Strategy: GLOBAL_REWRITE (Taxonomy completely wrong)")
    elif score < 6.0:
        diagnosis["fix_strategy"] = "ENTITY_REFINE"
        diagnosis["fix_severity"] = "major"
        log("  [Step4] Fix Strategy: ENTITY_REFINE (Wrong subtype/variant)")
    elif score < 8.0:
        diagnosis["fix_strategy"] = "DETAIL_ENHANCE"
        diagnosis["fix_severity"] = "minor"
        log("  [Step4] Fix Strategy: DETAIL_ENHANCE (Correct entity, needs detail)")
    else:
        diagnosis["fix_strategy"] = "ACCEPT"
        diagnosis["fix_severity"] = "none"
        log("  [Step4] Fix Strategy: ACCEPT (Excellent quality)")

    return diagnosis


# ==============================================================================
# Step 5: Reflexive Re-generation (带外挂的重试)
# ==============================================================================
def reflexive_regeneration(pipe, diagnosis, interpretation,
                           ref_image_path, dino_extractor,
                           current_dino_lambda, retry_idx,
                           seed, omnigen_device, output_path,
                           base_negative_prompt="",
                           height=1024, width=1024,
                           img_guidance_scale=1.6,
                           text_guidance_scale=2.5,
                           dino_lambda_step=0.15,
                           dino_lambda_max=0.8,
                           decouple_threshold=0.25,
                           dino_ref_path=None,
                           f_log=None):
    """
    Stage 5: Reflexive Re-generation with escalating DINO constraint.

    Key mechanisms:
    1. Negative prompt injection: Add TAC-identified errors to negative prompt
    2. DINO λ escalation: Increase structural prior weight each retry
    3. Refined prompt: Use TAC's refined_prompt for better text guidance

    Returns:
        next_dino_lambda: Updated DINO lambda for potential next retry
    """
    log = lambda msg: (f_log.write(msg + "\n") if f_log else print(msg))

    # 1. Escalate DINO λ
    next_dino_lambda = min(current_dino_lambda + dino_lambda_step, dino_lambda_max)
    log(f"  [Step5] DINO λ escalation: {current_dino_lambda:.2f} → {next_dino_lambda:.2f}")

    # 2. Build enhanced negative prompt
    tac_negative = diagnosis.get("refined_negative_prompt", "")
    critique = diagnosis.get("critique", "")

    neg_parts = set(x.strip() for x in base_negative_prompt.split(",") if x.strip())

    # Inject TAC-identified negatives
    if tac_negative:
        for part in tac_negative.split(","):
            if part.strip():
                neg_parts.add(part.strip())

    # Auto-generate negatives from error analysis
    fix_strategy = diagnosis.get("fix_strategy", "")
    if fix_strategy == "GLOBAL_REWRITE":
        neg_parts.update(["wrong subject", "incorrect object", "different category"])
    elif fix_strategy == "ENTITY_REFINE":
        neg_parts.update(["wrong variant", "incorrect model", "generic version"])

    enhanced_negative = ", ".join(sorted(neg_parts))
    log(f"  [Step5] Enhanced negative prompt: {enhanced_negative[:200]}")

    # 3. Get refined prompt from TAC
    refined_prompt = diagnosis.get("refined_prompt", interpretation.get("generation_prompt", ""))
    log(f"  [Step5] Refined prompt: {refined_prompt[:200]}...")

    # 4. Re-generate with DINO-injected parameters
    effective_guidance = dino_injected_generation(
        pipe=pipe,
        generation_prompt=refined_prompt,
        ref_image_path=ref_image_path,
        dino_extractor=dino_extractor,
        dino_lambda=next_dino_lambda,
        seed=seed + retry_idx + 1,
        omnigen_device=omnigen_device,
        output_path=output_path,
        f_log=f_log,
        height=height,
        width=width,
        img_guidance_scale=img_guidance_scale,
        text_guidance_scale=text_guidance_scale,
        negative_prompt=enhanced_negative,
        decouple_threshold=decouple_threshold,
        dino_ref_path=dino_ref_path,
    )

    return next_dino_lambda


# ==============================================================================
# Helper Functions (VRAM Management)
# ==============================================================================
GLOBAL_QWEN_MODEL = None
GLOBAL_QWEN_PROCESSOR = None
retriever = None
dino_extractor_global = None
retrieval_device = "cuda"


def move_helpers_to_cpu():
    """Aggressively move helper models (VLM, Retrieval, DINO) to CPU to free VRAM for OmniGen."""
    global GLOBAL_QWEN_MODEL, retriever, dino_extractor_global

    if args.vlm_device_id is None or args.vlm_device_id == args.device_id:
        print("[Memory] Moving helpers to CPU...")
        if GLOBAL_QWEN_MODEL is not None:
            try:
                GLOBAL_QWEN_MODEL.to("cpu")
            except:
                pass
        if retriever is not None:
            try:
                if hasattr(retriever, 'model') and retriever.model is not None:
                    retriever.model.to("cpu")
            except:
                pass
        if dino_extractor_global is not None:
            try:
                dino_extractor_global.to("cpu")
            except:
                pass
        gc.collect()
        torch.cuda.empty_cache()


def move_helpers_to_gpu(target_device):
    """Move helper models back to GPU for Retrieval/Critic tasks."""
    global GLOBAL_QWEN_MODEL, retriever, dino_extractor_global

    if args.vlm_device_id is None or args.vlm_device_id == args.device_id:
        print(f"[Memory] Moving helpers back to {target_device}...")
        gc.collect()
        torch.cuda.empty_cache()
        if GLOBAL_QWEN_MODEL is not None:
            try:
                GLOBAL_QWEN_MODEL.to(target_device)
            except:
                pass
        if retriever is not None:
            try:
                if hasattr(retriever, 'model') and retriever.model is not None:
                    retriever.model.to(target_device)
            except:
                pass
        if dino_extractor_global is not None:
            try:
                dino_extractor_global.to(target_device)
            except:
                pass
        torch.cuda.empty_cache()


# ==============================================================================
# Dataset Loading
# ==============================================================================
def load_db(use_nobg=False):
    """Load retrieval database images. Returns a dict of ds_name -> paths instead of flat list."""
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
    return dataset_splits


# ==============================================================================
# System Setup
# ==============================================================================
def setup_system(omnigen_device, vlm_device_map,
                 shared_qwen_model=None, shared_qwen_processor=None):
    """Initialize OmniGen2 pipeline and VLM client."""
    sys.path.append(os.path.abspath(args.omnigen2_path))

    try:
        from src.utils.custom_pipeline import CustomOmniGen2DiTLateFusionPipeline
        pipe = CustomOmniGen2DiTLateFusionPipeline.from_pretrained(
            args.omnigen2_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Acceleration
        if args.enable_teacache and args.enable_taylorseer:
            print("WARNING: TeaCache and TaylorSeer are mutually exclusive. Ignoring TeaCache.")
            args.enable_teacache = False

        if args.enable_taylorseer:
            pipe.enable_taylorseer = True
            if hasattr(pipe.transformer, "enable_teacache"):
                pipe.transformer.enable_teacache = False
        elif args.enable_teacache:
            if hasattr(pipe.transformer, "enable_teacache"):
                pipe.transformer.enable_teacache = True
                pipe.transformer.rel_l1_thresh = args.teacache_thresh
        else:
            if hasattr(pipe.transformer, "enable_teacache"):
                pipe.transformer.enable_teacache = False
            pipe.enable_taylorseer = False

        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        if args.enable_offload:
            pipe.enable_model_cpu_offload(device=omnigen_device)
        else:
            pipe.to(omnigen_device)

    except ImportError as e:
        print(f"Error: OmniGen2 not found: {e}")
        sys.exit(1)

    # VLM Client
    if not args.openai_api_key:
        print(f"[Setup] Using Local Qwen3-VL from {args.local_model_weight_path}")
        client = LocalQwen3VLWrapper(
            args.local_model_weight_path,
            device_map=vlm_device_map,
            shared_model=shared_qwen_model,
            shared_processor=shared_qwen_processor,
        )
        args.llm_model = "local-qwen3-vl"
    else:
        print("[Setup] Using SiliconFlow API...")
        client = openai.OpenAI(
            api_key=args.openai_api_key,
            base_url="https://api.siliconflow.cn/v1/",
        )

    client = UsageTrackingClient(client)
    return pipe, client


# ==============================================================================
# Main Pipeline
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()
    seed_everything(args.seed)

    # --- Configuration ---
    dt = datetime.now()
    timestamp = dt.strftime("%Y.%-m.%-d")
    run_time = dt.strftime("%H-%M-%S")
    _rm = args.retrieval_method

    DATASET_CONFIG = {
        "classes_txt": "datasets/fgvc-aircraft-2013b/data/variants.txt",
        "train_list": "datasets/fgvc-aircraft-2013b/data/images_train.txt",
        "image_root": "datasets/fgvc-aircraft-2013b/data/images",
        "output_path": f"results/{_rm}/{timestamp}/OmniGenV2_TAC_DINO_Importance_Aircraft_{run_time}",
    }

    # --- Load Retrieval Database ---
    retrieval_db = load_db(use_nobg=args.use_nobg)

    # --- Device Configuration ---
    retrieval_device = "cuda"
    if args.retrieval_cpu:
        retrieval_device = "cpu"
    elif isinstance(vlm_device_map, dict) and "" in vlm_device_map:
        retrieval_device = vlm_device_map[""]
    elif isinstance(vlm_device_map, str) and vlm_device_map != "auto":
        retrieval_device = vlm_device_map
    print(f"[Main] Retrieval/VLM Device: {retrieval_device}")

    # --- Load Shared Qwen3-VL ---
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        print("[Main] Loading shared Qwen3-VL for VAR + TAC...")
        GLOBAL_QWEN_PROCESSOR = AutoProcessor.from_pretrained(
            args.local_model_weight_path, trust_remote_code=True
        )

        quantization_config = None
        if args.qwen_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        GLOBAL_QWEN_MODEL = AutoModelForImageTextToText.from_pretrained(
            args.local_model_weight_path,
            torch_dtype=torch.bfloat16,
            device_map=vlm_device_map,
            trust_remote_code=True,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        ).eval()

    except Exception as e:
        print(f"Error loading Qwen3-VL: {e}")

    # --- Initialize Retriever ---
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
        print(f"Warning: Retriever init failed: {e}")

    # --- Initialize DINOv3 Feature Extractor ---
    try:
        print("[Main] Initializing DINOv3 Feature Extractor...")
        dino_extractor_global = DINOv3FeatureExtractor(
            model_path=args.dino_model_path,
            weights_path=args.dino_weights_path,
            device=retrieval_device,
        )
    except Exception as e:
        print(f"Warning: DINOv3 init failed: {e}")
        dino_extractor_global = None

    # --- Initialize Generator Pipeline ---
    pipe, client = setup_system(
        omnigen_device, vlm_device_map,
        shared_qwen_model=GLOBAL_QWEN_MODEL,
        shared_qwen_processor=GLOBAL_QWEN_PROCESSOR,
    )

    # --- Prepare Output Directories ---
    os.makedirs(DATASET_CONFIG['output_path'], exist_ok=True)
    logs_dir = os.path.join(DATASET_CONFIG['output_path'], "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # --- Resource Monitoring ---
    monitor = ResourceMonitor(interval=1.0)
    monitor.start()

    # --- Save Run Config ---
    config_path = os.path.join(logs_dir, "run_config.txt")
    with open(config_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Importance-Based 5-Step Pipeline Configuration\n")
        f.write("=" * 60 + "\n\n")
        f.write("[Command Line Arguments]\n")
        for arg in sorted(vars(args)):
            f.write(f"  {arg}: {getattr(args, arg)}\n")
        f.write(f"\n[Runtime Info]\n")
        f.write(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}\n")
        f.write(f"  OmniGen Device: {omnigen_device}\n")
        f.write(f"  Retrieval Device: {retrieval_device}\n")
        f.write(f"  DINO λ_init={args.dino_lambda_init}, λ_step={args.dino_lambda_step}, λ_max={args.dino_lambda_max}\n")

    # --- Load Class List ---
    with open(DATASET_CONFIG['classes_txt']) as f:
        all_classes = [l.strip() for l in f.readlines() if l.strip()]

    my_tasks = [c for i, c in enumerate(all_classes) if i % args.total_chunks == args.task_index]
    print(f"[Main] Processing {len(my_tasks)} classes out of {len(all_classes)} total.")

    # ==========================================================================
    # Main Loop: 5-Step Pipeline per Class
    # ==========================================================================
    for class_idx, class_name in enumerate(tqdm(my_tasks, desc="5-Step Pipeline")):
        safe_name = class_name.replace(" ", "_").replace("/", "-")
        prompt = f"a photo of a {class_name}"

        log_file = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}.log")
        f_log = open(log_file, "w")
        f_log.write(f"{'=' * 60}\n")
        f_log.write(f"Class: {class_name} ({class_idx + 1}/{len(my_tasks)})\n")
        f_log.write(f"Prompt: {prompt}\n")
        f_log.write(f"{'=' * 60}\n\n")

        try:
            # ==================================================================
            # STEP 1: Input Interpreter (非对称解析 / Importance Decomposition)
            # ==================================================================
            f_log.write(">>> STEP 1: Input Interpreter (Importance Decomposition)\n")
            move_helpers_to_gpu(retrieval_device)

            interpretation = importance_aware_input_interpreter(
                prompt, client, args.llm_model, domain="aircraft"
            )

            retrieval_query = interpretation.get("retrieval_query", prompt)
            generation_prompt = interpretation.get("generation_prompt", prompt)
            hi = interpretation.get("high_importance", {})
            lo = interpretation.get("low_importance", {})
            weights = interpretation.get("importance_weights", {})

            f_log.write(f"  HIGH_IMPORTANCE (entity): {hi.get('entity', 'N/A')}\n")
            f_log.write(f"  HIGH_IMPORTANCE (features): {hi.get('key_features', [])}\n")
            f_log.write(f"  LOW_IMPORTANCE: bg={lo.get('background', 'N/A')}, "
                        f"style={lo.get('visual_style', 'N/A')}\n")
            f_log.write(f"  WEIGHTS: {weights}\n")
            f_log.write(f"  Retrieval Query: {retrieval_query}\n")
            f_log.write(f"  Generation Prompt: {generation_prompt[:200]}...\n\n")

            # Generate Knowledge Specs for TAC
            reference_specs = None
            try:
                reference_specs = generate_knowledge_specs(
                    class_name, client, args.llm_model, domain="aircraft"
                )
                f_log.write(f"  Knowledge Specs: {reference_specs[:200]}...\n\n")
            except Exception as e:
                f_log.write(f"  Knowledge Specs generation failed: {e}\n\n")

            # ==================================================================
            # STEP 2: Dual-Stage Retrieval (护城河过滤)
            # ==================================================================
            f_log.write(">>> STEP 2: Dual-Stage Retrieval (Moat Filtering)\n")

            valid_refs, best_ref_score, var_details = dual_stage_retrieval(
                retrieval_query=retrieval_query,
                retriever=retriever,
                client=client,
                model_name=args.llm_model,
                var_k=args.var_k,
                f_log=f_log,
            )

            best_ref = valid_refs[0] if valid_refs else None

            f_log.write(f"  Best Reference: {os.path.basename(best_ref) if best_ref else 'None'}\n")
            f_log.write(f"  VAR Stats: passed={var_details.get('stage_b_passed', 0)}/"
                        f"{var_details.get('stage_a_count', 0)}\n\n")

            # ------------------------------------------------------------------
            # [新增] STEP 2.5: SAM2 Physical Matting (物理隔离)
            # ------------------------------------------------------------------
            dino_ref_image = best_ref  # 默认 DINO 也用原图

            if args.use_sam2_matting and best_ref is not None:
                f_log.write(">>> STEP 2.5: SAM2 Physical Matting for DINO\n")
                clean_ref_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_clean_ref.png")
                dino_ref_image = apply_sam2_matting(best_ref, clean_ref_path, device=retrieval_device)
                f_log.write(f"  Clean ref for DINO saved to: {dino_ref_image}\n\n")

            # ==================================================================
            # STEP 3: Initial Generation (DINO-Injected / 结构先验注入)
            # ==================================================================
            f_log.write(">>> STEP 3: Initial Generation (DINO Structure Prior Injection)\n")

            v1_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V1.png")
            current_dino_lambda = args.dino_lambda_init

            effective_guidance = dino_injected_generation(
                pipe=pipe,
                generation_prompt=generation_prompt,
                ref_image_path=best_ref,         # 👈 给 OmniGen2 依然传入带背景的原图！
                dino_ref_path=dino_ref_image,    # 👈 [新增传参] 给 DINO 传入抠黑背景的图！
                dino_extractor=dino_extractor_global,
                dino_lambda=current_dino_lambda,
                seed=args.seed,
                omnigen_device=omnigen_device,
                output_path=v1_path,
                f_log=f_log,
                height=args.height,
                width=args.width,
                img_guidance_scale=args.image_guidance_scale,
                text_guidance_scale=args.text_guidance_scale,
                negative_prompt=args.negative_prompt,
                decouple_threshold=args.decouple_threshold,
            )
            f_log.write(f"  V1 generated: {v1_path}\n\n")

            # ==================================================================
            # STEP 4: TAC Spatial Diagnosis (空间级诊断)
            # ==================================================================
            f_log.write(">>> STEP 4: TAC Spatial Diagnosis\n")

            current_image = v1_path
            best_score = -1
            best_image_path = None
            retry_cnt = 0

            diagnosis = tac_spatial_diagnosis(
                prompt=prompt,
                image_path=current_image,
                client=client,
                model_name=args.llm_model,
                reference_specs=reference_specs,
                domain="aircraft",
                dino_extractor=dino_extractor_global,
                ref_image_path=best_ref,
                f_log=f_log,
            )

            score = diagnosis.get("final_score", 0)
            if score > best_score:
                best_score = score
                best_image_path = current_image

            f_log.write(f"  Initial Score: {score}\n")
            f_log.write(f"  Fix Strategy: {diagnosis.get('fix_strategy', 'N/A')}\n\n")

            # ==================================================================
            # STEP 5: Reflexive Re-generation Loop (带外挂的重试)
            # ==================================================================
            if score < args.tac_pass_threshold:
                f_log.write(f">>> STEP 5: Reflexive Re-generation "
                            f"(score {score} < threshold {args.tac_pass_threshold})\n")

                while retry_cnt < args.max_retries:
                    retry_cnt += 1
                    f_log.write(f"\n  --- Retry {retry_cnt}/{args.max_retries} ---\n")

                    next_path = os.path.join(
                        DATASET_CONFIG['output_path'],
                        f"{safe_name}_V{retry_cnt + 1}.png"
                    )

                    # -------------------------------------------------------------
                    # [新增] 智能换胎逻辑 (Reference Image Rotation)
                    # -------------------------------------------------------------
                    current_ref = best_ref
                    current_dino_ref = dino_ref_image

                    if valid_refs and retry_cnt < len(valid_refs):
                        backup_ref = valid_refs[retry_cnt]
                        f_log.write(f"  [Rotation] Using backup reference image: {os.path.basename(backup_ref)}\n")
                        current_ref = backup_ref
                        current_dino_ref = backup_ref # 默认也是用原图

                        if args.use_sam2_matting:
                            f_log.write(f"  [Rotation] Applying SAM2 Physical Matting to backup reference...\n")
                            clean_backup_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_retry{retry_cnt}_clean_ref.png")
                            clean_dino = apply_sam2_matting(backup_ref, clean_backup_path, device=retrieval_device)
                            if clean_dino:
                                current_dino_ref = clean_dino
                                f_log.write(f"  [Rotation] Clean backup ref saved to: {clean_backup_path}\n")
                    # -------------------------------------------------------------

                    # Step 5: Reflexive Re-generation
                    current_dino_lambda = reflexive_regeneration(
                        pipe=pipe,
                        diagnosis=diagnosis,
                        interpretation=interpretation,
                        ref_image_path=current_ref,
                        dino_extractor=dino_extractor_global,
                        current_dino_lambda=current_dino_lambda,
                        retry_idx=retry_cnt,
                        seed=args.seed,
                        omnigen_device=omnigen_device,
                        output_path=next_path,
                        base_negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        img_guidance_scale=args.image_guidance_scale,
                        text_guidance_scale=args.text_guidance_scale,
                        dino_lambda_step=args.dino_lambda_step,
                        dino_lambda_max=args.dino_lambda_max,
                        decouple_threshold=args.decouple_threshold,
                        dino_ref_path=current_dino_ref,
                        f_log=f_log,
                    )
                    current_image = next_path

                    # Re-diagnose (Step 4 again)
                    f_log.write(f"\n  Re-Diagnosis after retry {retry_cnt}:\n")
                    diagnosis = tac_spatial_diagnosis(
                        prompt=prompt,
                        image_path=current_image,
                        client=client,
                        model_name=args.llm_model,
                        reference_specs=reference_specs,
                        domain="aircraft",
                        dino_extractor=dino_extractor_global,
                        ref_image_path=best_ref,
                        f_log=f_log,
                    )

                    score = diagnosis.get("final_score", 0)
                    f_log.write(f"  Retry {retry_cnt} Score: {score}\n")

                    if score > best_score:
                        best_score = score
                        best_image_path = current_image

                    # Early stop conditions
                    if score >= args.tac_early_stop_threshold:
                        f_log.write(f"  >> EARLY STOP: Excellent quality (score={score} >= {args.tac_early_stop_threshold})\n")
                        break
                    elif score >= args.tac_pass_threshold and diagnosis.get("taxonomy_check") == "correct":
                        f_log.write(f"  >> PASS: Taxonomy correct, score={score} >= {args.tac_pass_threshold}\n")
                        break

            else:
                f_log.write(f"  >> PASS on first attempt (score={score} >= {args.tac_pass_threshold})\n")

            # ==================================================================
            # Save Final Result
            # ==================================================================
            final_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_FINAL.png")
            if best_image_path and os.path.exists(best_image_path):
                shutil.copy(best_image_path, final_path)
                f_log.write(f"\n>>> FINAL: {os.path.basename(best_image_path)} → FINAL "
                            f"(score={best_score})\n")
            else:
                f_log.write(f"\n>>> WARNING: No valid image generated for {class_name}\n")

            # Save per-class summary JSON
            summary = {
                "class_name": class_name,
                "prompt": prompt,
                "retrieval_query": retrieval_query,
                "generation_prompt": generation_prompt[:500],
                "importance_weights": weights,
                "best_ref": os.path.basename(best_ref) if best_ref else None,
                "best_ref_score": best_ref_score,
                "var_passed": var_details.get("stage_b_passed", 0),
                "var_total": var_details.get("stage_a_count", 0),
                "initial_score": diagnosis.get("final_score", 0) if retry_cnt == 0 else "N/A",
                "final_score": best_score,
                "total_retries": retry_cnt,
                "dino_lambda_final": current_dino_lambda,
                "dino_similarity": diagnosis.get("dino_similarity", 0.0),
                "fix_strategy": diagnosis.get("fix_strategy", "N/A"),
            }
            summary_path = os.path.join(logs_dir, f"{safe_name}_summary.json")
            with open(summary_path, "w") as sf:
                json.dump(summary, sf, indent=2, ensure_ascii=False)

        except Exception as e:
            f_log.write(f"\n>>> EXCEPTION: {e}\n")
            import traceback
            f_log.write(traceback.format_exc())
            print(f"[ERROR] Class '{class_name}': {e}")

        finally:
            f_log.close()

    # ==========================================================================
    # Cleanup
    # ==========================================================================
    monitor.stop()
    monitor.save_plots(logs_dir)

    end_time = time.time()
    elapsed = end_time - start_time

    # Final Summary
    final_log_path = os.path.join(logs_dir, "pipeline_summary.txt")
    with open(final_log_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Importance-Based 5-Step Pipeline — Run Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total execution time: {elapsed:.2f}s ({elapsed / 60:.1f}min)\n")
        f.write(f"Classes processed: {len(my_tasks)}\n")
        f.write(f"Retrieval method: {args.retrieval_method}\n")
        f.write(f"DINO config: λ_init={args.dino_lambda_init}, "
                f"λ_step={args.dino_lambda_step}, λ_max={args.dino_lambda_max}\n")
        f.write(f"TAC thresholds: pass={args.tac_pass_threshold}, "
                f"early_stop={args.tac_early_stop_threshold}\n")
        f.write(f"\nToken Usage:\n")
        f.write(f"  Input Tokens:  {RUN_STATS['input_tokens']}\n")
        f.write(f"  Output Tokens: {RUN_STATS['output_tokens']}\n")
        f.write(f"  Total Tokens:  {RUN_STATS['input_tokens'] + RUN_STATS['output_tokens']}\n")

    print(f"\n[Done] Pipeline complete in {elapsed:.2f}s.")
    print(f"[Done] Results saved to: {DATASET_CONFIG['output_path']}")
    print(f"[Done] Token usage: {RUN_STATS['input_tokens'] + RUN_STATS['output_tokens']} total")
