"""
OmniGenV2_TAC_DINO_Importance_Aircraft_AR_slim.py
==================================================
精简 5-Step Pipeline — 双 Client 架构:
        text_client  → Qwen/Qwen3-Omni-30B-A3B-Instruct (SiliconFlow API)
                                 用于: Step1 input_interpreter + generate_knowledge_specs
    vl_client    → 本地 Qwen3-VL-4B
                                 用于: Step2 VAR + Step4/5 属性式 TAC（多个 yes/no 问题）

Retry=1 逻辑:
    retry=0: 属性式 TAC 按属性问题计算 yes/no 准确率，达阈值→ACCEPT，否则进入 retry=1
    retry=1: 大模型 generation_prompt 直接生成, 跳过 critic, ACCEPT (ground truth)
"""

from datetime import datetime
import argparse, sys, os

from _result_metadata import collect_run_metadata, write_run_metadata_block, write_run_metadata_json

# ==============================================================================
# Argument Parsing
# ==============================================================================
parser = argparse.ArgumentParser(description="OmniGenV2 DINO Importance Pipeline (slim, dual-client)")

# Devices
parser.add_argument("--device_id",         type=str, required=True)
parser.add_argument("--vlm_device_id",     type=str, default=None)
parser.add_argument("--task_index",        type=int, default=0)
parser.add_argument("--total_chunks",      type=int, default=1)
parser.add_argument("--warmup_n_classes",  type=int, default=0)

# Paths
parser.add_argument("--omnigen2_path",       type=str, default="./OmniGen2")
parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2")

# Text API (Qwen3-Omni-30B)
parser.add_argument("--text_api_key",   type=str, default=None,
                    help="SiliconFlow API key for text_client")
parser.add_argument("--text_model",     type=str,
                    default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                    help="Model for Step1 (pure text reasoning)")
parser.add_argument("--text_api_base",  type=str,
                    default="https://api.siliconflow.cn/v1/")
parser.add_argument("--text_max_tokens",type=int, default=8192)

# VL client (local Qwen3-VL or API)
parser.add_argument("--vl_api_key",              type=str, default=None,
                    help="If set, use API for VL; else use local model")
parser.add_argument("--vl_llm_model",            type=str,
                    default="Qwen/Qwen3-VL-30B-A3B-Instruct")
parser.add_argument("--local_model_weight_path", type=str,
                    default="/home/tingyu/imageRAG/Qwen3-VL-4B-Instruct")

# Generation
parser.add_argument("--seed",                 type=int,   default=0)
parser.add_argument("--max_retries",          type=int,   default=1,
                    help="Max retries; retry=1 uses LLM output as ground truth")
parser.add_argument("--height",               type=int,   default=1024)
parser.add_argument("--width",                type=int,   default=1024)
parser.add_argument("--image_guidance_scale", type=float, default=1.6)
parser.add_argument("--text_guidance_scale",  type=float, default=2.5)
parser.add_argument("--decouple_threshold",   type=float, default=0.25)

# Acceleration
parser.add_argument("--enable_offload",      action="store_true", default=True)
parser.add_argument("--disable_offload",     action="store_false", dest="enable_offload")
parser.add_argument("--enable_taylorseer",   action="store_true", default=True)
parser.add_argument("--disable_taylorseer",  action="store_false", dest="enable_taylorseer")

# Retrieval
parser.add_argument("--embeddings_path",    type=str,   default="datasets/embeddings/aircraft")
parser.add_argument("--retrieval_method",   type=str,   default="LongCLIP",
                    choices=["CLIP","LongCLIP","SigLIP","SigLIP2","Qwen2.5-VL","Qwen3-VL"])
parser.add_argument("--use_hybrid_retrieval", action="store_true")
parser.add_argument("--retrieval_datasets", nargs="+",  default=["aircraft"])
parser.add_argument("--var_k",              type=int,   default=10)
parser.add_argument("--retrieval_cpu",      action="store_true")

# DINO
parser.add_argument("--dino_model_path",   type=str, default="dinov3/")
parser.add_argument("--dino_weights_path", type=str,
                    default="dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
parser.add_argument("--dino_lambda_init",  type=float, default=0.3)
parser.add_argument("--dino_lambda_step",  type=float, default=0.15)
parser.add_argument("--dino_lambda_max",   type=float, default=0.8)

# Thresholds (attribute-TAC: score = yes_ratio × 10)
parser.add_argument("--tac_pass_threshold",       type=float, default=6.0)
parser.add_argument("--tac_early_stop_threshold", type=float, default=8.0)

args = parser.parse_args()

# ==============================================================================
# Device Setup
# ==============================================================================
if args.vlm_device_id and args.vlm_device_id != args.device_id:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device_id},{args.vlm_device_id}"
    omnigen_device = "cuda:0"
    vlm_device_map = {"": "cuda:1"}
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    omnigen_device = "cuda:0"
    vlm_device_map = "auto"

print(f"[Config] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

# ==============================================================================
# Imports
# ==============================================================================
import gc, json, re, shutil, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import openai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.critical.taxonomy_aware_critic import (
    generate_knowledge_specs,
    message_gpt,
    encode_image,
    _extract_json_candidate,
    _parse_json_loose,
)
from src.critical.binary_critic import retrieval_caption_generation as binary_critic_call
from src.retrieval.memory_guided_retrieval import ImageRetriever
from src.utils.rag_utils import (
    LocalQwen3VLWrapper,
    UsageTrackingClient,
    ResourceMonitor,
    RUN_STATS,
    seed_everything,
)

# ==============================================================================
# Text API Helpers (Qwen3-Omni-30B)
# ==============================================================================
TEXT_JSON_SYSTEM_PROMPT = "You are a helpful assistant designed to output JSON."


def get_text_response(resp):
    """优先提取 content，若为空则回退到 reasoning_content。"""
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        content = (getattr(resp.choices[0].message, "reasoning_content", None) or "").strip()
    return content


def call_text_api(text_client, messages, max_tokens=None, json_mode=False):
    """调用 text_client；结构化场景可启用 JSON mode。"""
    max_tokens = max_tokens or args.text_max_tokens
    req_messages = list(messages)
    if json_mode:
        req_messages = [{"role": "system", "content": TEXT_JSON_SYSTEM_PROMPT}] + req_messages

    kwargs = {
        "model": args.text_model,
        "messages": req_messages,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = text_client.chat.completions.create(**kwargs)
    result = get_text_response(resp)
    # 累计 token 统计
    if hasattr(resp, 'usage') and resp.usage:
        try:
            RUN_STATS['input_tokens']  += resp.usage.prompt_tokens
            RUN_STATS['output_tokens'] += resp.usage.completion_tokens
        except: pass
    return result


# ==============================================================================
# DINOv3 Feature Extractor
# ==============================================================================
class DINOv3FeatureExtractor:
    def __init__(self, model_path, weights_path, device="cuda"):
        self.device = device
        self.model  = None
        self.transform = None
        self._load_model(model_path, weights_path)

    def _load_model(self, model_path, weights_path):
        try:
            import types
            sys.path.append(model_path)
            mock_data     = types.ModuleType("dinov3.data")
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
                ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
                state_dict = ckpt.get('model', ckpt.get('teacher', ckpt))
                new_sd = {k.replace("module.", "").replace("_orig_mod.", ""): v
                          for k, v in state_dict.items()}
                self.model.load_state_dict(new_sd, strict=False)
            self.model.eval().to(self.device)
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
            print("[DINO] Loaded successfully.")
        except Exception as e:
            print(f"[DINO] Load error: {e}")

    @torch.no_grad()
    def extract_patch_features(self, image_path):
        if self.model is None: return None, None
        try:
            img = Image.open(image_path).convert("RGB")
            t   = self.transform(img).unsqueeze(0).to(self.device)
            cls_token = self.model(t)
            if cls_token.dim() != 2: cls_token = cls_token[:, 0]
            return None, cls_token
        except Exception as e:
            print(f"[DINO] extract error: {e}"); return None, None

    @torch.no_grad()
    def compute_similarity(self, path_a, path_b):
        _, a = self.extract_patch_features(path_a)
        _, b = self.extract_patch_features(path_b)
        if a is None or b is None: return 0.0
        return F.cosine_similarity(a, b, dim=-1).item()

    def to(self, device):
        if self.model: self.model.to(device); self.device = str(device)
        return self


# ==============================================================================
# Step 1: Importance-Aware Input Interpreter  (uses text_client)
# ==============================================================================
def importance_aware_input_interpreter(prompt, text_client, domain="aircraft"):
    """大模型（Qwen3.5-397B, thinking on）拆解 prompt → JSON。"""
    domain_ctx = {"aircraft": "aircraft identification and aviation photography",
                  "birds": "ornithology and bird photography",
                  "cub":   "ornithology and bird photography"}.get(domain, "fine-grained visual recognition")

    msg = f"""You are an expert Input Interpreter Agent specialized in {domain_ctx}.
Decompose the following prompt for a RAG-based text-to-image pipeline.

**User Prompt:** "{prompt}"

Output ONLY one valid JSON object (no prose, no markdown):
{{
  "high_importance": {{"entity": "<exact subject name>", "key_features": ["<f1>","<f2>"]}},
  "low_importance":  {{"background": "...", "composition": "...", "lighting": "...", "visual_style": "..."}},
  "importance_weights": {{"entity_identity": 0.9, "structural_detail": 0.8, "environment": 0.3, "artistic_style": 0.2}},
  "retrieval_query": "a photo of a <entity>",
  "generation_prompt": "<full detailed prompt for T2I>",
  "ambiguous_elements": []
}}

STRICT: entity must be the exact subject name from prompt. generation_prompt combines all elements into one rich description.
"""
    import re
    def _ck(k):
        k = re.sub(r"[^a-z0-9]", "_", str(k).strip().lower())
        k = re.sub(r"_+","_",k).strip("_"); return k
    def _co(obj):
        if isinstance(obj,dict): return {_ck(k):_co(v) for k,v in obj.items()}
        if isinstance(obj,list): return [_co(x) for x in obj]
        return obj
    def _pick(d, keys, default=None):
        if not isinstance(d,dict): return default
        for k in keys:
            if _ck(k) in d: return d[_ck(k)]
        return default
    def _default():
        ent = prompt.replace("a photo of a ","").replace("a photo of ","").strip() or prompt
        return {"high_importance":{"entity":ent,"key_features":[]},
                "low_importance":{"background":"clear sky","composition":"side view",
                                  "lighting":"natural daylight","visual_style":"photorealistic"},
                "importance_weights":{"entity_identity":0.9,"structural_detail":0.8,
                                      "environment":0.3,"artistic_style":0.2},
                "retrieval_query":f"a photo of a {ent}",
                "generation_prompt":prompt,"ambiguous_elements":[],
                "_parser_recovery_used":False}

    for attempt in range(2):
        try:
            ans = call_text_api(text_client,
                                [{"role":"user","content":msg}],
                                json_mode=True)
            parsed = _parse_json_loose(_extract_json_candidate(ans))
            if not isinstance(parsed, dict): raise ValueError("not a dict")
            root = _co(parsed)
            hi   = _co(_pick(root,["high_importance","identified_elements"],{}))
            lo   = _co(_pick(root,["low_importance","creativity_fills"],{}))
            iw   = _co(_pick(root,["importance_weights","weights"],{}))
            entity = str(_pick(hi,["entity","main_subject","subject"],"") or "").strip()
            if not entity:
                entity = prompt.replace("a photo of a ","").strip() or prompt
            lo_def = {"background":"clear sky","composition":"side view",
                      "lighting":"natural daylight","visual_style":"photorealistic"}
            lo_out = {k: str(_pick(lo,[k],None) or v).strip() or v for k,v in lo_def.items()}
            iw_def = {"entity_identity":0.9,"structural_detail":0.8,
                      "environment":0.3,"artistic_style":0.2}
            iw_out = {k: float(v) for k,v in {
                k: _pick(iw,[k],v) for k,v in iw_def.items()}.items()}
            rq = _pick(root,["retrieval_query","query"], f"a photo of a {entity}")
            gp = _pick(root,["generation_prompt","detailed_prompt","prompt"], prompt)
            result = {
                "high_importance":{"entity":entity,"key_features":
                    [str(x) for x in (_pick(hi,["key_features","features"],[]) or []) if str(x).strip()]},
                "low_importance":lo_out, "importance_weights":iw_out,
                "retrieval_query":str(rq).strip(),
                "generation_prompt":str(gp).strip(),
                "ambiguous_elements":[],
                "_parser_recovery_used":False,
            }
            if result["high_importance"]["entity"]:
                return result
        except Exception as e:
            print(f"[InputInterpreter] attempt {attempt}: {e}")

    res = _default(); res["_parser_recovery_used"] = True
    return res


def build_rule_based_interpretation(prompt):
    entity = prompt.replace("a photo of a ", "").replace("a photo of ", "").strip() or prompt
    return {
        "high_importance": {"entity": entity, "key_features": []},
        "low_importance": {
            "background": "clear sky",
            "composition": "side view",
            "lighting": "natural daylight",
            "visual_style": "photorealistic",
        },
        "importance_weights": {
            "entity_identity": 0.9,
            "structural_detail": 0.8,
            "environment": 0.3,
            "artistic_style": 0.2,
        },
        "retrieval_query": f"a photo of a {entity}",
        "generation_prompt": prompt,
        "ambiguous_elements": [],
        "_parser_recovery_used": False,
    }


# ==============================================================================
# Step 2: Dual-Stage Retrieval  (uses vl_client)
# ==============================================================================
def dual_stage_retrieval(retrieval_query, retriever, vl_client, vl_model,
                          var_k=10, f_log=None):
    log = lambda m: (f_log.write(m+"\n") if f_log else print(m))
    log(f"  [Step2-A] Base Retrieval: '{retrieval_query}' K={var_k}")
    try:
        lists, scores = retriever.search([retrieval_query])
        if not lists or not lists[0]: return [], 0.0, {"stage_a_count":0,"stage_b_passed":0}
        candidates = lists[0][:var_k]; cscores = scores[0][:var_k]
        log(f"  [Step2-A] {len(candidates)} candidates.")
    except Exception as e:
        log(f"  [Step2-A] Error: {e}"); return [], 0.0, {"error":str(e)}

    entity = retrieval_query.replace("a photo of a ","").replace("a photo of ","").strip()
    valid_refs, best_score = [], 0.0
    var_details = {"stage_a_count":len(candidates),"stage_b_passed":0,"stage_b_results":[]}

    for idx,(cp,cs) in enumerate(zip(candidates,cscores)):
        try:
            img64 = encode_image(cp)
            if img64 is None: continue
            resp = vl_client.chat.completions.create(
                model=vl_model,
                messages=[{"role":"user","content":[
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img64}"}},
                    {"type":"text","text":f"Does this image show '{entity}'? Answer YES or NO only."}
                ]}],
                temperature=0.01, max_tokens=8,
            )
            ans = (resp.choices[0].message.content or "").strip().upper()
            passed = "YES" in ans
            var_details["stage_b_results"].append({"path":os.path.basename(cp),"passed":passed})
            if passed:
                var_details["stage_b_passed"] += 1
                valid_refs.append(cp)
                if len(valid_refs)==1: best_score = cs
                log(f"    [#{idx}] PASS ✓ {os.path.basename(cp)}")
            else:
                log(f"    [#{idx}] FAIL ✗ ({ans[:40]})")
        except Exception as e:
            log(f"    [#{idx}] ERROR: {e}")

    log(f"  [Step2] passed={var_details['stage_b_passed']}/{len(candidates)}")
    return valid_refs, best_score, var_details


# ==============================================================================
# Step 3: DINO-Injected Generation
# ==============================================================================
def dino_injected_generation(pipe, generation_prompt, ref_image_path,
                              dino_extractor, dino_lambda, seed,
                              output_path, f_log=None,
                              height=1024, width=1024,
                              img_guidance=1.6, text_guidance=2.5,
                              decouple_threshold=0.25):
    log = lambda m: (f_log.write(m+"\n") if f_log else print(m))
    eff_img = img_guidance
    if dino_extractor and ref_image_path:
        try:
            _, cls = dino_extractor.extract_patch_features(ref_image_path)
            if cls is not None:
                conf = min(cls.norm(dim=-1).mean().item()/20.0, 1.0)
                eff_img = min(img_guidance*(1.0+dino_lambda*conf), 3.0)
                log(f"  [Step3-DINO] λ={dino_lambda:.2f} conf={conf:.3f} img_g={eff_img:.3f}")
        except Exception as e:
            log(f"  [Step3-DINO] error: {e}")

    gen_prompt = (f"{generation_prompt}. Use reference image <|image_1|> for structural guidance."
                  if ref_image_path else generation_prompt)
    move_helpers_to_cpu()

    input_images = []
    if ref_image_path and os.path.exists(ref_image_path):
        input_images.append(Image.open(ref_image_path).convert("RGB"))

    gen = torch.Generator("cuda").manual_seed(seed)
    try:
        res = pipe(prompt=gen_prompt,
                   input_images=input_images or None,
                   height=height, width=width,
                   text_guidance_scale=text_guidance,
                   image_guidance_scale=eff_img,
                   num_inference_steps=50,
                   generator=gen,
                   decouple_threshold=decouple_threshold)
        res.images[0].save(output_path)
        log(f"  [Step3] Saved {output_path}")
    except Exception as e:
        log(f"  [Step3] Error: {e}. Fallback to text-only.")
        try:
            gen = torch.Generator("cuda").manual_seed(seed)
            res = pipe(prompt=generation_prompt, input_images=None,
                       height=height, width=width,
                       text_guidance_scale=text_guidance, image_guidance_scale=1.0,
                       num_inference_steps=50, generator=gen,
                       decouple_threshold=decouple_threshold)
            res.images[0].save(output_path)
            log(f"  [Step3] Fallback saved.")
        except Exception as e2:
            log(f"  [Step3] Fallback failed: {e2}")
    return eff_img


# ==============================================================================
# Step 4: Attribute-based TAC  (uses vl_client, multi-question YES/NO)
# ==============================================================================
def _clean_feature_text(text):
    text = re.sub(r"^\s*[-*•]+\s*", "", str(text or "")).strip()
    text = re.sub(r"^\s*\d+[\.)]\s*", "", text).strip()
    text = text.strip(" \t:-")
    return text


def collect_tac_features(key_features=None, reference_specs=None, max_items=5):
    features, seen = [], set()

    def add_feature(raw):
        clean = _clean_feature_text(raw)
        if not clean:
            return
        norm = re.sub(r"\s+", " ", clean).strip().lower()
        if not norm or len(norm) < 3:
            return
        if norm.startswith((
            "hard constraints",
            "focus on",
            "output format",
            "output as",
            "task:",
            "you are",
        )):
            return
        if norm in seen:
            return
        seen.add(norm)
        features.append(clean)

    for feat in key_features or []:
        add_feature(feat)

    if reference_specs:
        chunks = []
        for line in str(reference_specs).splitlines():
            line = line.strip()
            if line:
                chunks.append(line)
        if len(chunks) <= 1:
            chunks = [x.strip() for x in re.split(r"[;\n]", str(reference_specs)) if x.strip()]
        for chunk in chunks:
            add_feature(chunk)

    return features[:max_items]


def generate_vlm_reference_specs(class_name, image_path, vl_client, vl_model, f_log=None, max_items=4):
    log = lambda m: (f_log.write(m+"\n") if f_log else print(m))
    if not image_path or not os.path.exists(image_path):
        log("  [VLMAttr] No reference image available.")
        return ""

    move_helpers_to_gpu(retrieval_device)
    img64 = encode_image(image_path)
    if img64 is None:
        log("  [VLMAttr] Cannot encode reference image.")
        return ""

    try:
        resp = vl_client.chat.completions.create(
            model=vl_model,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img64}"}},
                {"type": "text", "text":
                    f"You are an aviation expert. Inspect this reference image of '{class_name}'. "
                    f"List the top {max_items} visual identification features that are clearly visible and useful for distinguishing this aircraft variant. "
                    "Focus on engine count/placement, wing configuration, tail, fuselage shape, nose, winglets, and other distinguishing structure. "
                    "Output only short bullet points, one per line."}
            ]}],
            temperature=0.01,
            max_tokens=196,
        )
        specs = (resp.choices[0].message.content or "").strip()
        log(f"  [VLMAttr] Specs[:200]: {specs[:200]}")
        return specs
    except Exception as e:
        log(f"  [VLMAttr] Error: {e}")
        return ""


def binary_critic_diagnosis(prompt, image_path, vl_client, vl_model,
                            entity_name=None, key_features=None, reference_specs=None,
                            dino_extractor=None, ref_image_path=None, f_log=None):
    log = lambda m: (f_log.write(m+"\n") if f_log else print(m))
    move_helpers_to_gpu(retrieval_device)

    result = binary_critic_call(prompt, [image_path], vl_client, vl_model)
    accepted = result.get("status") == "success"
    score = 8.0 if accepted else 2.0
    accuracy = 1.0 if accepted else 0.0
    checks = [{"question": "Q1", "label": "global_identity", "present": accepted}]
    missing_attributes = [] if accepted else ["global_identity"]
    log(f"  [BinaryCritic] verdict={'YES/ACCEPT' if accepted else 'NO/REJECT'} score={score}")

    dino_sim = 0.0
    if dino_extractor and ref_image_path:
        try:
            dino_sim = dino_extractor.compute_similarity(image_path, ref_image_path)
            log(f"  [Step4-DINO] sim={dino_sim:.4f}")
        except Exception as e:
            log(f"  [Step4-DINO] error: {e}")

    return {
        "status": result.get("status", "error"),
        "final_score": score,
        "accepted": accepted,
        "attribute_accuracy": accuracy,
        "yes_count": int(accepted),
        "total_questions": 1,
        "attribute_checks": checks,
        "missing_attributes": missing_attributes,
        "binary_critic_result": result,
        "dino_similarity": dino_sim,
        "fix_strategy": "ACCEPT" if accepted else "GLOBAL_REWRITE",
        "refined_prompt": prompt,
        "critique": "Global binary yes/no critic.",
        "taxonomy_check": "correct" if accepted else "wrong_subtype",
    }


def tac_attribute_diagnosis(prompt, image_path, vl_client, vl_model,
                            entity_name=None, key_features=None, reference_specs=None,
                            dino_extractor=None, ref_image_path=None, f_log=None):
    log = lambda m: (f_log.write(m+"\n") if f_log else print(m))
    move_helpers_to_gpu(retrieval_device)

    target_entity = (entity_name or "").strip()
    if not target_entity:
        target_entity = prompt.replace("a photo of a ", "").replace("a photo of ", "").strip() or prompt

    feature_list = collect_tac_features(key_features=key_features, reference_specs=reference_specs, max_items=5)
    questions = [
        ("entity_identity", f"Does the image depict the target aircraft '{target_entity}' rather than a different aircraft type or variant?")
    ]
    questions.extend(
        (feat, f"Does the image clearly show this target attribute: '{feat}'?")
        for feat in feature_list
    )

    img64 = encode_image(image_path)
    if img64 is None:
        log("  [Step4-TAC] Cannot encode image.")
        return {
            "status": "error",
            "final_score": 0.0,
            "accepted": False,
            "attribute_accuracy": 0.0,
            "yes_count": 0,
            "total_questions": len(questions),
            "attribute_checks": [],
            "missing_attributes": [q[0] for q in questions],
            "dino_similarity": 0.0,
            "fix_strategy": "GLOBAL_REWRITE",
            "refined_prompt": prompt,
            "critique": "image encoding failed",
            "taxonomy_check": "error",
        }

    q_text = "\n".join(f"Q{i+1}: {question}" for i, (_, question) in enumerate(questions))
    max_tokens = max(64, 12 * len(questions))

    try:
        resp = vl_client.chat.completions.create(
            model=vl_model,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img64}"}},
                {"type": "text", "text":
                    "You are an aircraft verification judge. Carefully inspect the image and answer every question using YES or NO only.\n"
                    f"{q_text}\n"
                    "Reply ONLY in this format:\nQ1: YES/NO\nQ2: YES/NO\n..."}
            ]}],
            temperature=0.01,
            max_tokens=max_tokens,
        )
        ans_text = (resp.choices[0].message.content or "").upper()
    except Exception as e:
        log(f"  [Step4-TAC] VLM error: {e}")
        ans_text = ""

    log(f"  [Step4-TAC] Raw answer: {ans_text[:240]}")

    checks = []
    yes_count = 0
    for i, (label, _) in enumerate(questions):
        match = re.search(rf"Q{i+1}[:\s-]+([A-Z]+)", ans_text)
        is_yes = bool(match and "YES" in match.group(1))
        checks.append({"question": f"Q{i+1}", "label": label, "present": is_yes})
        yes_count += int(is_yes)
        log(f"    [Q{i+1}] {'YES' if is_yes else 'NO '} | {label}")

    total_questions = len(checks)
    accuracy = yes_count / total_questions if total_questions else 0.0
    score = round(accuracy * 10.0, 2)
    identity_ok = checks[0]["present"] if checks else False
    accepted = identity_ok and score >= args.tac_pass_threshold
    missing_attributes = [c["label"] for c in checks if not c["present"]]

    if not identity_ok:
        taxonomy_check = "wrong_subtype"
    elif accepted:
        taxonomy_check = "correct"
    else:
        taxonomy_check = "wrong_subtype"

    if score >= args.tac_early_stop_threshold and accepted:
        fix_strategy = "ACCEPT"
    elif identity_ok:
        fix_strategy = "DETAIL_ENHANCE"
    else:
        fix_strategy = "GLOBAL_REWRITE"

    critique = (
        "All TAC attribute checks passed."
        if not missing_attributes else
        f"Missing or unclear attributes: {missing_attributes}"
    )
    log(f"  [Step4-TAC] yes={yes_count}/{total_questions} accuracy={accuracy:.2%} score={score:.2f} accepted={accepted}")

    dino_sim = 0.0
    if dino_extractor and ref_image_path:
        try:
            dino_sim = dino_extractor.compute_similarity(image_path, ref_image_path)
            log(f"  [Step4-DINO] sim={dino_sim:.4f}")
        except Exception as e:
            log(f"  [Step4-DINO] error: {e}")

    return {
        "status": "success" if accepted else "error",
        "final_score": score,
        "accepted": accepted,
        "attribute_accuracy": accuracy,
        "yes_count": yes_count,
        "total_questions": total_questions,
        "attribute_checks": checks,
        "missing_attributes": missing_attributes,
        "dino_similarity": dino_sim,
        "fix_strategy": fix_strategy,
        "refined_prompt": prompt,
        "critique": critique,
        "taxonomy_check": taxonomy_check,
    }


# ==============================================================================
# Step 4b: TIFA Feature-Level Evaluation  (uses vl_client, one YES/NO per feature)
#   仅在 max_retries > 2 时启用
# ==============================================================================
def tifa_feature_eval(key_features, image_path, vl_client, vl_model, f_log=None):
    """
    对 key_features 列表中的每个特征逐条询问 VLM YES/NO。
    返回: {feature_str: bool}  (True=present, False=missing)
    """
    log = lambda m: (f_log.write(m+"\n") if f_log else print(m))
    if not key_features:
        return {}

    move_helpers_to_gpu(retrieval_device)
    results = {}
    try:
        img64 = encode_image(image_path)
        if img64 is None:
            log("  [TIFA] Cannot encode image, skipping feature eval.")
            return {f: False for f in key_features}

        # 将所有特征合并成一次 VLM 调用（节省时间）
        questions = "\n".join(
            f"Q{i+1}: Does the image show '{feat}'? Answer YES or NO."
            for i, feat in enumerate(key_features)
        )
        resp = vl_client.chat.completions.create(
            model=vl_model,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img64}"}},
                {"type": "text", "text":
                    f"Carefully examine this image and answer each question with YES or NO.\n{questions}\n"
                    f"Reply ONLY in format: Q1: YES/NO, Q2: YES/NO, ..."}
            ]}],
            temperature=0.01, max_tokens=64,
        )
        ans_text = (resp.choices[0].message.content or "").upper()
        log(f"  [TIFA] VLM answer: {ans_text[:200]}")

        import re
        for i, feat in enumerate(key_features):
            # 匹配 Q{i+1}: YES 或 Q{i+1}: NO
            m = re.search(rf"Q{i+1}[:\s]+([A-Z]+)", ans_text)
            present = bool(m and "YES" in m.group(1))
            results[feat] = present
            log(f"    {'✓' if present else '✗'} {feat}")
    except Exception as e:
        log(f"  [TIFA] Error: {e}")
        results = {f: False for f in key_features}

    return results


# ==============================================================================
# Step 5a: LLM Prompt Reinforcement  (uses text_client)
#   接收缺失特征列表 → LLM 重写 generation_prompt，强化缺失属性
#   仅在 max_retries > 2 时启用
# ==============================================================================
def llm_reinforce_prompt(gen_prompt, failed_features, text_client, entity_name, f_log=None):
    """
    让 LLM 基于当前 gen_prompt 和缺失特征列表，输出强化版 prompt。
    只输出新 prompt，不输出任何解释。
    """
    log = lambda m: (f_log.write(m+"\n") if f_log else print(m))
    if not failed_features:
        return gen_prompt

    failed_str = "; ".join(f"'{f}'" for f in failed_features)
    msg = f"""You are a text-to-image prompt engineer.

Original prompt:
"{gen_prompt}"

The generated image is MISSING these visual features: {failed_str}.

Rewrite the prompt to STRONGLY EMPHASIZE the missing features using:
- Repetition of key terms
- Descriptive adjectives (e.g., "clearly visible", "prominently featuring", "must show")
- Moving missing features to the BEGINNING of the prompt

Output ONLY the new prompt text. No explanation, no quotes, no markdown."""

    try:
        new_prompt = call_text_api(text_client, [{"role": "user", "content": msg}],
                                    max_tokens=512)
        new_prompt = new_prompt.strip().strip('"').strip("'").strip()
        if not new_prompt or len(new_prompt) < 20:
            log(f"  [LLMReinforce] Empty/short response, keep original prompt.")
            return gen_prompt
        log(f"  [LLMReinforce] New prompt[:200]: {new_prompt[:200]}")
        return new_prompt
    except Exception as e:
        log(f"  [LLMReinforce] Error: {e}, keep original.")
        return gen_prompt


def rule_reinforce_prompt(gen_prompt, failed_features, entity_name, f_log=None):
    log = lambda m: (f_log.write(m+"\n") if f_log else print(m))
    clean_features = [str(f).strip() for f in (failed_features or []) if str(f).strip() and str(f).strip() != "entity_identity"]
    if not clean_features:
        return gen_prompt

    lead = f"a photo of a {entity_name}, clearly showing " if entity_name else "clearly showing "
    emphasis = ", ".join(f"{feat}" for feat in clean_features)
    new_prompt = f"{lead}{emphasis}. {gen_prompt}".strip()
    log(f"  [RuleReinforce] New prompt[:200]: {new_prompt[:200]}")
    return new_prompt


# ==============================================================================
# Step 5: Reflexive Re-generation
# ==============================================================================
def reflexive_regeneration(pipe, generation_prompt, ref_image_path,
                            dino_extractor, current_lambda, retry_idx,
                            seed, output_path,
                            height=1024, width=1024,
                            img_guidance=1.6, text_guidance=2.5,
                            dino_lambda_step=0.15, dino_lambda_max=0.8,
                            decouple_threshold=0.25, f_log=None):
    log = lambda m: (f_log.write(m+"\n") if f_log else print(m))
    next_lambda = min(current_lambda + dino_lambda_step, dino_lambda_max)
    log(f"  [Step5] λ: {current_lambda:.2f}→{next_lambda:.2f}, prompt[:100]={generation_prompt[:100]}")
    dino_injected_generation(
        pipe=pipe, generation_prompt=generation_prompt,
        ref_image_path=ref_image_path, dino_extractor=dino_extractor,
        dino_lambda=next_lambda, seed=seed+retry_idx+1,
        output_path=output_path, f_log=f_log,
        height=height, width=width,
        img_guidance=img_guidance, text_guidance=text_guidance,
        decouple_threshold=decouple_threshold,
    )
    return next_lambda


# ==============================================================================
# VRAM Management
# ==============================================================================
GLOBAL_QWEN_MODEL     = None
GLOBAL_QWEN_PROCESSOR = None
retriever             = None
dino_extractor_global = None
retrieval_device      = "cuda"


def move_helpers_to_cpu():
    if args.vlm_device_id and args.vlm_device_id != args.device_id: return
    for obj in [GLOBAL_QWEN_MODEL,
                getattr(retriever,'model',None) if retriever else None,
                dino_extractor_global]:
        if obj is not None:
            try: obj.to("cpu")
            except: pass
    gc.collect(); torch.cuda.empty_cache()


def move_helpers_to_gpu(target):
    if args.vlm_device_id and args.vlm_device_id != args.device_id: return
    gc.collect(); torch.cuda.empty_cache()
    for obj in [GLOBAL_QWEN_MODEL,
                getattr(retriever,'model',None) if retriever else None,
                dino_extractor_global]:
        if obj is not None:
            try: obj.to(target)
            except: pass
    torch.cuda.empty_cache()


# ==============================================================================
# Dataset Loading
# ==============================================================================
def load_db():
    dataset_splits = {}
    for ds in args.retrieval_datasets:
        paths = []
        if ds == 'aircraft':
            root = "datasets/fgvc-aircraft-2013b/data/images"
            lf   = "datasets/fgvc-aircraft-2013b/data/images_train.txt"
            if os.path.exists(lf):
                with open(lf) as f:
                    paths = [os.path.join(root, l.strip()+".jpg") for l in f if l.strip()]
        elif ds == 'cub':
            root = "datasets/CUB_200_2011/images"
            sf = "datasets/CUB_200_2011/train_test_split.txt"
            im = "datasets/CUB_200_2011/images.txt"
            if os.path.exists(sf) and os.path.exists(im):
                train_ids = {l.split()[0] for l in open(sf) if len(l.split())>=2 and l.split()[1]=='1'}
                paths = [os.path.join(root,l.split()[1]) for l in open(im)
                         if len(l.split())>=2 and l.split()[0] in train_ids]
        elif ds == 'imagenet':
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            imagenet_list_candidates = [
                os.path.join(repo_root, 'datasets/imagenet_train_list.txt'),
                'datasets/imagenet_train_list.txt',
            ]
            imagenet_root_candidates = [
                os.path.join(repo_root, 'datasets/ILSVRC2012_train'),
                '/home/tingyu/imageRAG/datasets/ILSVRC2012_train',
                os.path.join(repo_root, 'datasets/imagenet/train'),
                os.path.join(repo_root, 'datasets/ILSVRC/Data/CLS-LOC/train'),
                os.path.join(repo_root, 'datasets/imagenet'),
                'datasets/ILSVRC2012_train',
                'datasets/imagenet/train',
                'datasets/ILSVRC/Data/CLS-LOC/train',
                'datasets/imagenet',
            ]

            imagenet_root = next((p for p in imagenet_root_candidates if os.path.isdir(p)), None)
            imagenet_list = next((p for p in imagenet_list_candidates if os.path.isfile(p)), None)

            if imagenet_root and imagenet_list:
                with open(imagenet_list, 'r', encoding='utf-8') as f:
                    paths = [os.path.join(imagenet_root, l.strip()) for l in f if l.strip()]
                print(f"  imagenet: loaded full train list from {imagenet_list}")

            import glob
            for root_candidate in imagenet_root_candidates:
                if paths:
                    break
                if os.path.isdir(root_candidate):
                    paths = glob.glob(os.path.join(root_candidate, "**", "*.JPEG"), recursive=True)
                    if not paths:
                        paths = glob.glob(os.path.join(root_candidate, "**", "*.jpg"), recursive=True)
                    if paths:
                        break
            if not paths:
                # Fallback: read paths from existing embedding cache
                emb_dir = "datasets/embeddings/imagenet"
                method_key = args.retrieval_method.lower().replace("-", "")
                import glob as _gl
                pts = sorted(_gl.glob(os.path.join(emb_dir, f"*embeddings_b*.pt")))
                if pts:
                    dummy = torch.load(pts[0], map_location="cpu", weights_only=False)
                    paths = dummy.get("paths", [])
                    print(f"  imagenet: loaded {len(paths)} paths from embedding cache ({os.path.basename(pts[0])})")
                else:
                    print(f"  [WARNING] imagenet: no images or embeddings found!")
        dataset_splits[ds] = paths
        print(f"  {ds}: {len(paths)} paths")
    return dataset_splits



# ==============================================================================
# System Setup
# ==============================================================================
def setup_system(omnigen_device, vlm_device_map,
                 shared_qwen_model=None, shared_qwen_processor=None):
    # -- Pipeline --
    sys.path.append(os.path.abspath(args.omnigen2_path))
    try:
        from src.utils.custom_pipeline import CustomOmniGen2DualPathPipeline
        pipe = CustomOmniGen2DualPathPipeline.from_pretrained(
            args.omnigen2_model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, local_files_only=True)
        pipe.enable_taylorseer = args.enable_taylorseer
        if hasattr(pipe.transformer,"enable_teacache"):
            pipe.transformer.enable_teacache = False
        pipe.vae.enable_tiling(); pipe.vae.enable_slicing()
        if args.enable_offload: pipe.enable_model_cpu_offload(device=omnigen_device)
        else: pipe.to(omnigen_device)
    except Exception as e:
        print(f"Error loading OmniGen2: {e}"); sys.exit(1)

    # -- Text Client (Qwen3-Omni-30B) --
    if args.text_api_key:
        _tc = openai.OpenAI(api_key=args.text_api_key, base_url=args.text_api_base)
        text_client = UsageTrackingClient(_tc)
        print(f"[Setup] text_client → {args.text_model} (SiliconFlow API, JSON mode for Step1)")
    else:
        print("[Setup] WARNING: --text_api_key not set. Using fallback local VL for text tasks.")
        text_client = None  # Will be set to vl_client after

    # -- VL Client (local Qwen3-VL or API) --
    if args.vl_api_key:
        _vc = openai.OpenAI(api_key=args.vl_api_key, base_url=args.text_api_base)
        vl_client = UsageTrackingClient(_vc)
        vl_model  = args.vl_llm_model
        print(f"[Setup] vl_client → {vl_model} (API)")
    else:
        _vc = LocalQwen3VLWrapper(
            args.local_model_weight_path, device_map=vlm_device_map,
            shared_model=shared_qwen_model, shared_processor=shared_qwen_processor)
        vl_client = UsageTrackingClient(_vc)
        vl_model  = "local-qwen3-vl"
        print(f"[Setup] vl_client → local Qwen3-VL ({args.local_model_weight_path})")

    if text_client is None:
        text_client = vl_client   # fallback
    return pipe, text_client, vl_client, vl_model


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()
    seed_everything(args.seed)

    dt        = datetime.now()
    timestamp = dt.strftime("%Y.%-m.%-d")
    run_time  = dt.strftime("%H-%M-%S")
    out_root  = f"results/{args.retrieval_method}/{timestamp}/OmniGenV2_TACAttr_AR_{run_time}"

    print(f"[Main] Output: {out_root}")
    retrieval_db = load_db()

    retrieval_device = "cpu" if args.retrieval_cpu else "cuda"

    # Load shared Qwen3-VL
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        print("[Main] Loading shared Qwen3-VL...")
        GLOBAL_QWEN_PROCESSOR = AutoProcessor.from_pretrained(
            args.local_model_weight_path, trust_remote_code=True)
        GLOBAL_QWEN_MODEL = AutoModelForImageTextToText.from_pretrained(
            args.local_model_weight_path, torch_dtype=torch.bfloat16,
            device_map=vlm_device_map, trust_remote_code=True, low_cpu_mem_usage=True).eval()
    except Exception as e:
        print(f"[Main] Could not load Qwen3-VL: {e}")

    # Multi-dataset retriever
    class MultiDatasetRetriever:
        def __init__(self, splits, method, device, k, hybrid, ext_model, ext_proc):
            self.k = k
            self.subs = [
                ImageRetriever(image_paths=paths,
                               embeddings_path=f"datasets/embeddings/{ds}",
                               method=method, device=device, k=k,
                               use_hybrid=hybrid,
                               external_model=ext_model, external_processor=ext_proc)
                for ds, paths in splits.items()
            ]
        def search(self, queries):
            all_p, all_s = [], []
            for r in self.subs:
                p, s = r.search(queries)
                all_p.extend(p); all_s.extend(s)
            if not all_p: return [], []
            combined = sorted(zip(all_p, all_s), key=lambda x:x[1], reverse=True)[:self.k]
            return [x[0] for x in combined], [x[1] for x in combined]

    is_qwen_retriever = args.retrieval_method in ["Qwen3-VL","Qwen2.5-VL"]
    retriever = MultiDatasetRetriever(
        splits=retrieval_db, method=args.retrieval_method,
        device=retrieval_device, k=args.var_k,
        hybrid=args.use_hybrid_retrieval,
        ext_model=GLOBAL_QWEN_MODEL if is_qwen_retriever else None,
        ext_proc=GLOBAL_QWEN_PROCESSOR if is_qwen_retriever else None,
    )
    torch.cuda.empty_cache()

    # DINOv3
    try:
        dino_extractor_global = DINOv3FeatureExtractor(
            args.dino_model_path, args.dino_weights_path, device=retrieval_device)
    except Exception as e:
        print(f"[Main] DINO init failed: {e}"); dino_extractor_global = None

    pipe, text_client, vl_client, vl_model = setup_system(
        omnigen_device, vlm_device_map,
        shared_qwen_model=GLOBAL_QWEN_MODEL,
        shared_qwen_processor=GLOBAL_QWEN_PROCESSOR,
    )

    os.makedirs(out_root, exist_ok=True)
    logs_dir = os.path.join(out_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    run_metadata = collect_run_metadata(__file__, args, result_dir=out_root, logs_dir=logs_dir)
    write_run_metadata_json(logs_dir, run_metadata)

    monitor = ResourceMonitor(interval=1.0); monitor.start()

    with open(os.path.join(logs_dir,"run_config.txt"),"w") as f:
        f.write("=== Dual-Client Attribute-TAC Pipeline ===\n")
        write_run_metadata_block(f, run_metadata)

    classes_txt = "datasets/fgvc-aircraft-2013b/data/variants.txt"
    with open(classes_txt) as f:
        all_classes = [l.strip() for l in f if l.strip()]
    my_tasks = [c for i,c in enumerate(all_classes) if i % args.total_chunks == args.task_index]
    if args.warmup_n_classes > 0:
        my_tasks = my_tasks[:args.warmup_n_classes]

    print(f"[Main] {len(my_tasks)}/{len(all_classes)} classes, max_retries={args.max_retries}")

    warmup = {"classes":0,"input_recovery":0,"binary_calls":0,"accepts":0,"scores":[]}

    # ===========================================================================
    # Main Loop
    # ===========================================================================
    for cls_idx, class_name in enumerate(tqdm(my_tasks, desc="Pipeline")):
        safe  = class_name.replace(" ","_").replace("/","-")
        prompt= f"a photo of a {class_name}"

        f_log = open(os.path.join(out_root,f"{safe}.log"),"w")
        f_log.write(f"{'='*60}\nClass: {class_name}\nPrompt: {prompt}\n{'='*60}\n\n")
        write_run_metadata_block(f_log, run_metadata)

        try:
            # ---- STEP 1: Input Interpreter (text_client) ----
            f_log.write(">>> STEP 1: Input Interpreter (LLM API)\n")
            move_helpers_to_gpu(retrieval_device)
            interp = importance_aware_input_interpreter(prompt, text_client, domain="aircraft")

            if args.warmup_n_classes > 0:
                warmup["classes"] += 1
                if interp.get("_parser_recovery_used"): warmup["input_recovery"] += 1

            entity    = interp["high_importance"]["entity"]
            ret_query = interp["retrieval_query"]
            gen_prompt= interp["generation_prompt"]   # 大模型 thinking 后的 ground truth prompt
            f_log.write(f"  Entity: {entity}\n  RetQuery: {ret_query}\n")
            f_log.write(f"  GenPrompt[:200]: {gen_prompt[:200]}\n\n")

            # Knowledge specs via text_client
            ref_specs = None
            try:
                # generate_knowledge_specs uses message_gpt internally; we replicate with text_client
                specs_msg = [{"role":"user","content":
                    f"You are an aviation expert. List the top 4 visual identification features of '{class_name}' (engine count/placement, wing config, tail, distinctive features). Plain bullet list only."}]
                ref_specs = call_text_api(text_client, specs_msg, max_tokens=512)
                f_log.write(f"  Specs[:150]: {ref_specs[:150]}\n\n")
            except Exception as e:
                f_log.write(f"  Specs error: {e}\n\n")

            # ---- STEP 2: Dual-Stage Retrieval (vl_client) ----
            f_log.write(">>> STEP 2: VAR Retrieval\n")
            valid_refs, best_score, var_details = dual_stage_retrieval(
                ret_query, retriever, vl_client, vl_model,
                var_k=args.var_k, f_log=f_log)
            best_ref = valid_refs[0] if valid_refs else None
            f_log.write(f"  best_ref: {os.path.basename(best_ref) if best_ref else 'None'}\n\n")

            # ---- STEP 3: Initial Generation ----
            f_log.write(">>> STEP 3: DINO-Injected Generation\n")
            v1_path = os.path.join(out_root, f"{safe}_V1.png")
            cur_lambda = args.dino_lambda_init
            dino_injected_generation(
                pipe=pipe, generation_prompt=gen_prompt,
                ref_image_path=best_ref, dino_extractor=dino_extractor_global,
                dino_lambda=cur_lambda, seed=args.seed, output_path=v1_path, f_log=f_log,
                height=args.height, width=args.width,
                img_guidance=args.image_guidance_scale, text_guidance=args.text_guidance_scale,
                decouple_threshold=args.decouple_threshold)

            # ---- STEP 4: Attribute-based TAC (vl_client) ----
            f_log.write(">>> STEP 4: Attribute-based TAC\n")
            best_image  = v1_path
            best_final_score = -1
            retry_cnt   = 0
            # TIFA 特征评估（仅 max_retries > 2 时执行）
            key_features = interp["high_importance"].get("key_features", [])
            critic_features = collect_tac_features(key_features=key_features, reference_specs=ref_specs, max_items=5)
            f_log.write(f"  TAC features: {critic_features}\n")
            tifa_use = args.max_retries > 2 and bool(key_features)
            feature_eval = {}   # 上一次各特征通过情况

            diagnosis = tac_attribute_diagnosis(
                prompt, v1_path, vl_client, vl_model,
                entity_name=entity, key_features=critic_features, reference_specs=None,
                dino_extractor=dino_extractor_global, ref_image_path=best_ref, f_log=f_log)

            if tifa_use:
                f_log.write("  [TIFA] Feature-level evaluation:\n")
                feature_eval = tifa_feature_eval(
                    key_features, v1_path, vl_client, vl_model, f_log=f_log)
                failed_features = [f for f, ok in feature_eval.items() if not ok]
                f_log.write(f"  [TIFA] failed={failed_features}\n")
            else:
                failed_features = []

            if args.warmup_n_classes > 0:
                warmup["binary_calls"] += 1
                if diagnosis["accepted"]: warmup["accepts"] += 1
                warmup["scores"].append(diagnosis["final_score"])

            accepted = diagnosis["accepted"]
            best_final_score = diagnosis["final_score"]
            current_gen_prompt = gen_prompt   # 当前生成使用的 prompt（可被 LLM 强化）
            if accepted: best_image = v1_path

            # ---- STEP 5: Retry (if not accepted) ----
            if not accepted and args.max_retries >= 1:
                f_log.write(f">>> STEP 5: Retry (max={args.max_retries}, TIFA={'ON' if tifa_use else 'OFF'})\n")
                while retry_cnt < args.max_retries:
                    retry_cnt += 1
                    next_path = os.path.join(out_root, f"{safe}_V{retry_cnt+1}.png")
                    cur_ref = valid_refs[retry_cnt] if (valid_refs and retry_cnt < len(valid_refs)) else best_ref

                    # ---- Step 5-A: LLM Prompt Reinforcement (仅 max_retries > 2) ----
                    if tifa_use and failed_features:
                        f_log.write(f"  [Step5-A] LLM prompt reinforcement, failed={failed_features}\n")
                        current_gen_prompt = llm_reinforce_prompt(
                            current_gen_prompt, failed_features, text_client,
                            entity_name=entity, f_log=f_log)

                    # ---- Step 5-B: Re-generate ----
                    cur_lambda = reflexive_regeneration(
                        pipe=pipe, generation_prompt=current_gen_prompt,
                        ref_image_path=cur_ref, dino_extractor=dino_extractor_global,
                        current_lambda=cur_lambda, retry_idx=retry_cnt,
                        seed=args.seed, output_path=next_path, f_log=f_log,
                        height=args.height, width=args.width,
                        img_guidance=args.image_guidance_scale, text_guidance=args.text_guidance_scale,
                        dino_lambda_step=args.dino_lambda_step, dino_lambda_max=args.dino_lambda_max,
                        decouple_threshold=args.decouple_threshold)

                    if retry_cnt >= args.max_retries:
                        # 末次重试 → 大模型 ground truth，直接 ACCEPT
                        f_log.write(f"  [Step5] retry={retry_cnt} = ground truth ACCEPT (no critic)\n")
                        best_image = next_path; best_final_score = 8.0
                        if args.warmup_n_classes > 0: warmup["accepts"] += 1
                        break

                    # ---- Step 5-C: Re-diagnose ----
                    f_log.write(f"\n  Re-diagnosis retry={retry_cnt}:\n")
                    diagnosis = tac_attribute_diagnosis(
                        prompt, next_path, vl_client, vl_model,
                        entity_name=entity, key_features=critic_features, reference_specs=None,
                        dino_extractor=dino_extractor_global, ref_image_path=best_ref, f_log=f_log)
                    if tifa_use:
                        feature_eval = tifa_feature_eval(
                            key_features, next_path, vl_client, vl_model, f_log=f_log)
                        failed_features = [f for f, ok in feature_eval.items() if not ok]
                        f_log.write(f"  [TIFA] re-eval failed={failed_features}\n")

                    if args.warmup_n_classes > 0:
                        warmup["binary_calls"] += 1
                        if diagnosis["accepted"]: warmup["accepts"] += 1
                        warmup["scores"].append(diagnosis["final_score"])
                    if diagnosis["accepted"]:
                        best_image = next_path; best_final_score = 8.0
                        f_log.write(f"  [Step5] ACCEPT at retry={retry_cnt}\n"); break
                    best_image = next_path  # keep latest even if rejected
            else:
                f_log.write(f"  [Step4] ACCEPT on first attempt.\n")

            # ---- Save FINAL ----
            final_path = os.path.join(out_root, f"{safe}_FINAL.png")
            if best_image and os.path.exists(best_image):
                shutil.copy(best_image, final_path)
            f_log.write(f"\n>>> FINAL: {os.path.basename(best_image)} score={best_final_score}\n")

            json.dump({
                "class_name": class_name, "prompt": prompt,
                "script_name": run_metadata["script_name"],
                "script_path": run_metadata["script_path"],
                "command_redacted": run_metadata["command_redacted"],
                "entity": entity, "retrieval_query": ret_query,
                "generation_prompt": gen_prompt[:500],
                "best_ref": os.path.basename(best_ref) if best_ref else None,
                "best_ref_score": best_score,
                "var_passed": var_details.get("stage_b_passed",0),
                "var_total":  var_details.get("stage_a_count",0),
                "final_score": best_final_score,
                "total_retries": retry_cnt,
                "dino_lambda_final": cur_lambda,
                "dino_similarity": diagnosis.get("dino_similarity",0.0),
                "tac_attribute_accuracy": diagnosis.get("attribute_accuracy", 0.0),
                "tac_yes_count": diagnosis.get("yes_count", 0),
                "tac_total_questions": diagnosis.get("total_questions", 0),
                "tac_missing_attributes": diagnosis.get("missing_attributes", []),
                "binary_critic_result": "accepted" if best_final_score >= 6.0 else "rejected",
            }, open(os.path.join(logs_dir,f"{safe}_summary.json"),"w"), indent=2, ensure_ascii=False)

        except Exception as e:
            import traceback
            f_log.write(f"\n>>> EXCEPTION: {e}\n{traceback.format_exc()}")
            print(f"[ERROR] {class_name}: {e}")
        finally:
            f_log.close()

    # ===========================================================================
    # Cleanup & Summary
    # ===========================================================================
    monitor.stop(); monitor.save_plots(logs_dir)
    elapsed = time.time() - start_time

    with open(os.path.join(logs_dir,"pipeline_summary.txt"),"w") as f:
        f.write("=== Dual-Client Attribute-TAC Pipeline Summary ===\n")
        write_run_metadata_block(f, run_metadata)
        f.write(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)\n")
        f.write(f"Classes: {len(my_tasks)}\n")
        f.write(f"text_model: {args.text_model}\n")
        f.write(f"vl_model:   {vl_model}\n")
        f.write(f"Tokens: in={RUN_STATS['input_tokens']}, out={RUN_STATS['output_tokens']}\n")
        if args.warmup_n_classes > 0:
            cc = max(1, warmup["classes"])
            bc = max(1, warmup["binary_calls"])
            f.write(f"\n[Warmup]\n")
            f.write(f"  Input recovery: {warmup['input_recovery']}/{cc} ({warmup['input_recovery']/cc:.2%})\n")
            f.write(f"  Attribute-TAC accept rate: {warmup['accepts']}/{bc} ({warmup['accepts']/bc:.2%})\n")
            avg = sum(warmup["scores"])/len(warmup["scores"]) if warmup["scores"] else 0.0
            f.write(f"  Avg score: {avg:.2f}\n")

    print(f"\n[Done] {elapsed:.1f}s | {out_root}")
    print(f"[Done] Tokens: {RUN_STATS['input_tokens']+RUN_STATS['output_tokens']} total")
