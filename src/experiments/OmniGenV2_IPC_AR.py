"""
OmniGenV2_IPC_AR.py
===================
Standalone IPC-AR main experiment.

核心组成：
- Step 1 Importance-aware input interpreter
- Step 2 Two-stage multimodal retrieval (LongCLIP recall + VAR rerank)
- Step 3 DINO-injected generation
- Step 4 IPC identity diagnosis
- Step 5 IPC-cue-driven LLM prompt reinforcement

说明：
- 这是论文主实验实现，不再依附于其他 experiment 文件。
- TIFA feature-level evaluation 不参与该主线，只保留 IPC 驱动的闭环语义修正。
"""

from datetime import datetime
import argparse, sys, os

from _result_metadata import collect_run_metadata, write_run_metadata_block, write_run_metadata_json


TEXT_LLM_PRESETS = {
	"qwen_omni": {
		"model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
		"base_url": "https://api.siliconflow.cn/v1/",
		"max_completion_tokens": 8192,
	},
	"kimi_k25": {
		"model": "Pro/moonshotai/Kimi-K2.5",
		"base_url": "https://api.siliconflow.cn/v1/",
		"max_completion_tokens": 8192,
	},
	"deepbricks_gpt51": {
		"model": "gpt-5.1",
		"base_url": "https://api.deepbricks.ai/v1/",
		"max_completion_tokens": 4096,
	},
}

# ==============================================================================
# Argument Parsing
# ==============================================================================
parser = argparse.ArgumentParser(description="OmniGenV2 IPC-AR Pipeline (standalone, dual-client)")

parser.add_argument("--device_id",         type=str, required=True)
parser.add_argument("--vlm_device_id",     type=str, default=None)
parser.add_argument("--task_index",        type=int, default=0)
parser.add_argument("--total_chunks",      type=int, default=1)
parser.add_argument("--warmup_n_classes",  type=int, default=0)

parser.add_argument("--omnigen2_path",       type=str, default="./OmniGen2")
parser.add_argument("--omnigen2_model_path", type=str, default="OmniGen2/OmniGen2")

parser.add_argument("--text_llm_preset",   type=str, default="qwen_omni",
					choices=["qwen_omni", "kimi_k25", "deepbricks_gpt51", "custom"],
					help="Preset for text LLM provider/model. Use custom to fully specify text_model + text_api_base.")
parser.add_argument("--text_api_key",    type=str, default=None)
parser.add_argument("--text_model",      type=str, default=None)
parser.add_argument("--text_api_base",   type=str, default=None)
parser.add_argument("--text_max_tokens", type=int, default=8192)

parser.add_argument("--vl_api_key",              type=str, default=None)
parser.add_argument("--vl_llm_model",            type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")
parser.add_argument("--local_model_weight_path", type=str,
					default="/home/tingyu/imageRAG/Qwen3-VL-4B-Instruct")

parser.add_argument("--seed",                 type=int,   default=0)
parser.add_argument("--max_retries",          type=int,   default=1)
parser.add_argument("--height",               type=int,   default=1024)
parser.add_argument("--width",                type=int,   default=1024)
parser.add_argument("--image_guidance_scale", type=float, default=1.6)
parser.add_argument("--text_guidance_scale",  type=float, default=2.5)
parser.add_argument("--decouple_threshold",   type=float, default=0.25)

parser.add_argument("--enable_offload",     action="store_true",  default=True)
parser.add_argument("--disable_offload",    action="store_false", dest="enable_offload")
parser.add_argument("--enable_taylorseer",  action="store_true",  default=True)
parser.add_argument("--disable_taylorseer", action="store_false", dest="enable_taylorseer")

parser.add_argument("--embeddings_path",    type=str,  default="datasets/embeddings/aircraft")
parser.add_argument("--retrieval_method",   type=str,  default="LongCLIP",
					choices=["CLIP","LongCLIP","SigLIP","SigLIP2","Qwen2.5-VL","Qwen3-VL"])
parser.add_argument("--use_hybrid_retrieval", action="store_true")
parser.add_argument("--retrieval_datasets", nargs="+", default=["aircraft"])
parser.add_argument("--var_k",              type=int,  default=10)
parser.add_argument("--retrieval_cpu",      action="store_true")
parser.add_argument("--siglip2_model_id",   type=str,  default=None,
					help="Optional HF model id for SigLIP2 retrieval, e.g. google/siglip2-so400m-patch16-naflex")

parser.add_argument("--dino_model_path",   type=str, default="dinov3/")
parser.add_argument("--dino_weights_path", type=str,
					default="dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
parser.add_argument("--dino_lambda_init",  type=float, default=0.3)
parser.add_argument("--dino_lambda_step",  type=float, default=0.15)
parser.add_argument("--dino_lambda_max",   type=float, default=0.8)

parser.add_argument("--tac_pass_threshold",       type=float, default=6.0)
parser.add_argument("--tac_early_stop_threshold", type=float, default=8.0)

args = parser.parse_args()

if args.text_llm_preset != "custom":
	preset = TEXT_LLM_PRESETS[args.text_llm_preset]
	if args.text_model is None:
		args.text_model = preset["model"]
	if args.text_api_base is None:
		args.text_api_base = preset["base_url"]
	max_cap = preset.get("max_completion_tokens")
	if max_cap is not None and args.text_max_tokens > max_cap:
		print(f"[Setup] text_max_tokens={args.text_max_tokens} exceeds preset limit {max_cap} for {args.text_llm_preset}; clamped.")
		args.text_max_tokens = max_cap
else:
	if args.text_model is None or args.text_api_base is None:
		parser.error("--text_llm_preset custom requires both --text_model and --text_api_base")

if args.siglip2_model_id:
	os.environ["SIGLIP2_MODEL_ID"] = args.siglip2_model_id

# ==============================================================================
# Device Setup
# ==============================================================================
if args.vlm_device_id and args.vlm_device_id != args.device_id:
	os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device_id},{args.vlm_device_id}"
	omnigen_device = "cuda:0"; vlm_device_map = {"": "cuda:1"}
else:
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
	omnigen_device = "cuda:0"; vlm_device_map = "auto"

# ==============================================================================
# Imports
# ==============================================================================
import gc, json, shutil, time
import torch, torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import openai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.critical.taxonomy_aware_critic import encode_image, _extract_json_candidate, _parse_json_loose
from src.critical.identity_preservation_critic import evaluate_identity_preservation
from src.retrieval.memory_guided_retrieval import ImageRetriever
from src.utils.rag_utils import (
	LocalQwen3VLWrapper, UsageTrackingClient, ResourceMonitor, RUN_STATS, seed_everything)

# ==============================================================================
# Text API Helpers
# ==============================================================================
TEXT_JSON_SYSTEM_PROMPT = "You are a helpful assistant designed to output JSON."
_TEXT_MAX_TOKENS_WARNED = False


def get_text_response(resp):
	content = (resp.choices[0].message.content or "").strip()
	if not content:
		content = (getattr(resp.choices[0].message, "reasoning_content", None) or "").strip()
	return content


def get_effective_text_max_tokens(requested_max_tokens=None):
	global _TEXT_MAX_TOKENS_WARNED
	requested = requested_max_tokens or args.text_max_tokens
	cap = None
	if args.text_llm_preset in TEXT_LLM_PRESETS:
		cap = TEXT_LLM_PRESETS[args.text_llm_preset].get("max_completion_tokens")
	if cap is not None and requested > cap:
		if not _TEXT_MAX_TOKENS_WARNED:
			print(f"[TextAPI] Requested max_tokens={requested} exceeds provider cap {cap}; using {cap}.")
			_TEXT_MAX_TOKENS_WARNED = True
		return cap
	return requested


def call_text_api(text_client, messages, max_tokens=None, json_mode=False):
	max_tokens = get_effective_text_max_tokens(max_tokens)
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
	if hasattr(resp, 'usage') and resp.usage:
		try:
			RUN_STATS['input_tokens']  += resp.usage.prompt_tokens
			RUN_STATS['output_tokens'] += resp.usage.completion_tokens
		except: pass
	return result

# ==============================================================================
# DINOv3
# ==============================================================================
class DINOv3FeatureExtractor:
	def __init__(self, model_path, weights_path, device="cuda"):
		self.device = device; self.model = None; self.transform = None
		try:
			import types
			sys.path.append(model_path)
			mock_data = types.ModuleType("dinov3.data")
			mock_ds   = types.ModuleType("dinov3.data.datasets")
			for cls_name in ["DatasetWithEnumeratedTargets","SamplerType","ImageDataAugmentation"]:
				setattr(mock_data, cls_name, type(cls_name,(object,),{}))
			mock_data.make_data_loader = lambda *a,**kw: None
			mock_data.datasets = mock_ds
			sys.modules.update({"dinov3.data":mock_data,"dinov3.data.datasets":mock_ds})

			self.model = torch.hub.load(model_path,"dinov3_vitb16",source="local",pretrained=False)
			if os.path.exists(weights_path):
				ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
				sd = ckpt.get('model', ckpt.get('teacher', ckpt))
				self.model.load_state_dict(
					{k.replace("module.","").replace("_orig_mod.",""): v for k,v in sd.items()}, strict=False)
			self.model.eval().to(device)
			from torchvision import transforms
			self.transform = transforms.Compose([
				transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
				transforms.CenterCrop(224), transforms.ToTensor(),
				transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
			])
			print("[DINO] Loaded.")
		except Exception as e:
			print(f"[DINO] Load error: {e}")

	@torch.no_grad()
	def extract_patch_features(self, image_path):
		if not self.model: return None, None
		try:
			t = self.transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
			cls = self.model(t)
			return None, (cls[:, 0] if cls.dim() != 2 else cls)
		except: return None, None

	@torch.no_grad()
	def compute_similarity(self, a, b):
		_, ca = self.extract_patch_features(a); _, cb = self.extract_patch_features(b)
		if ca is None or cb is None: return 0.0
		return F.cosine_similarity(ca, cb, dim=-1).item()

	def to(self, device):
		if self.model: self.model.to(device); self.device = str(device)
		return self

# ==============================================================================
# Step 1 (text_client)
# ==============================================================================
def importance_aware_input_interpreter(prompt, text_client, domain="aircraft"):
	if text_client is None:
		return build_rule_based_interpretation(prompt)
	import re
	def _ck(k):
		k = re.sub(r"[^a-z0-9]","_",str(k).strip().lower())
		return re.sub(r"_+","_",k).strip("_")
	def _co(obj):
		if isinstance(obj,dict): return {_ck(k):_co(v) for k,v in obj.items()}
		if isinstance(obj,list): return [_co(x) for x in obj]
		return obj
	def _pick(d,keys,default=None):
		if not isinstance(d,dict): return default
		for k in keys:
			if _ck(k) in d: return d[_ck(k)]
		return default
	def _default(recovery=False):
		ent = prompt.replace("a photo of a ","").strip() or prompt
		return {"high_importance":{"entity":ent,"key_features":[]},
				"low_importance":{"background":"clear sky","composition":"side view",
								  "lighting":"natural daylight","visual_style":"photorealistic"},
				"importance_weights":{"entity_identity":0.9,"structural_detail":0.8,
									  "environment":0.3,"artistic_style":0.2},
				"retrieval_query":f"a photo of a {ent}",
				"generation_prompt":prompt,"ambiguous_elements":[],
				"_parser_recovery_used":recovery}

	msg = f"""Expert Input Interpreter for {domain}.
Prompt: "{prompt}"
Output ONE valid JSON (no markdown):
{{"high_importance":{{"entity":"<exact name>","key_features":["f1","f2"]}},"low_importance":{{"background":"...","composition":"...","lighting":"...","visual_style":"..."}},"importance_weights":{{"entity_identity":0.9,"structural_detail":0.8,"environment":0.3,"artistic_style":0.2}},"retrieval_query":"a photo of a <entity>","generation_prompt":"<rich T2I prompt>","ambiguous_elements":[]}}"""

	for attempt in range(2):
		try:
			ans  = call_text_api(text_client, [{"role":"user","content":msg}], json_mode=True)
			root = _co(_parse_json_loose(_extract_json_candidate(ans)) or {})
			hi   = _co(_pick(root,["high_importance","identified_elements"],{}))
			lo   = _co(_pick(root,["low_importance","creativity_fills"],{}))
			iw   = _co(_pick(root,["importance_weights","weights"],{}))
			ent  = str(_pick(hi,["entity","main_subject"],"") or "").strip() or (
					   prompt.replace("a photo of a ","").strip())
			lo_def = {"background":"clear sky","composition":"side view","lighting":"natural daylight","visual_style":"photorealistic"}
			iw_def = {"entity_identity":0.9,"structural_detail":0.8,"environment":0.3,"artistic_style":0.2}
			return {
				"high_importance":{"entity":ent,"key_features":
					[str(x) for x in (_pick(hi,["key_features","features"],[]) or []) if str(x).strip()]},
				"low_importance":  {k:str(_pick(lo,[k],v) or v) for k,v in lo_def.items()},
				"importance_weights": {k:float(_pick(iw,[k],v) or v) for k,v in iw_def.items()},
				"retrieval_query":  str(_pick(root,["retrieval_query","query"],f"a photo of a {ent}")).strip(),
				"generation_prompt":str(_pick(root,["generation_prompt","detailed_prompt"],prompt)).strip(),
				"ambiguous_elements":[], "_parser_recovery_used":False,
			}
		except Exception as e:
			print(f"[InputInterpreter] attempt {attempt}: {e}")
	return _default(recovery=True)


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
		"_parser_recovery_used": True,
	}


# ==============================================================================
# Step 2 (vl_client, YES/NO VAR)
# ==============================================================================
def dual_stage_retrieval(retrieval_query, retriever, vl_client, vl_model, var_k=10, f_log=None):
	log = lambda m: (f_log.write(m+"\n") if f_log else print(m))
	try:
		lists, scores = retriever.search([retrieval_query])
		if not lists or not lists[0]: return [], 0.0, {"stage_a_count":0,"stage_b_passed":0}
		candidates, cscores = lists[0][:var_k], scores[0][:var_k]
	except Exception as e:
		log(f"  [Retrieval] error: {e}"); return [], 0.0, {"error":str(e)}

	entity = retrieval_query.replace("a photo of a ","").strip()
	valid_refs, best_score = [], 0.0
	details = {"stage_a_count":len(candidates),"stage_b_passed":0}

	for idx,(cp,cs) in enumerate(zip(candidates,cscores)):
		try:
			img64 = encode_image(cp)
			if not img64: continue
			resp = vl_client.chat.completions.create(
				model=vl_model,
				messages=[{"role":"user","content":[
					{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img64}"}},
					{"type":"text","text":f"Does this show '{entity}'? Answer YES or NO only."}]}],
				temperature=0.01, max_tokens=8)
			ans = (resp.choices[0].message.content or "").strip().upper()
			if "YES" in ans:
				details["stage_b_passed"] += 1
				valid_refs.append(cp)
				if len(valid_refs)==1: best_score = cs
				log(f"    [#{idx}] PASS ✓")
			else:
				log(f"    [#{idx}] FAIL ✗")
		except Exception as e:
			log(f"    [#{idx}] ERR: {e}")

	return valid_refs, best_score, details


# ==============================================================================
# Step 3: DINO-Injected Generation
# ==============================================================================
def dino_injected_generation(pipe, generation_prompt, ref_image_path,
							  dino_extractor, dino_lambda, seed, output_path, f_log=None,
							  height=1024, width=1024, img_guidance=1.6, text_guidance=2.5,
							  decouple_threshold=0.25):
	log = lambda m: (f_log.write(m+"\n") if f_log else print(m))
	eff_img = img_guidance
	if dino_extractor and ref_image_path:
		try:
			_, cls = dino_extractor.extract_patch_features(ref_image_path)
			if cls is not None:
				conf = min(cls.norm(dim=-1).mean().item()/20.0, 1.0)
				eff_img = min(img_guidance*(1.0+dino_lambda*conf), 3.0)
		except: pass

	gen_p = (f"{generation_prompt}. Use reference image <|image_1|>."
			 if ref_image_path else generation_prompt)
	move_helpers_to_cpu()
	imgs = ([Image.open(ref_image_path).convert("RGB")]
			if ref_image_path and os.path.exists(ref_image_path) else [])
	gen = torch.Generator("cuda").manual_seed(seed)
	try:
		r = pipe(prompt=gen_p, input_images=imgs or None, height=height, width=width,
				 text_guidance_scale=text_guidance, image_guidance_scale=eff_img,
				 num_inference_steps=50, generator=gen, decouple_threshold=decouple_threshold)
		r.images[0].save(output_path); log(f"  [Step3] Saved {output_path}")
	except Exception as e:
		log(f"  [Step3] Error: {e}. Fallback.")
		try:
			gen = torch.Generator("cuda").manual_seed(seed)
			r = pipe(prompt=generation_prompt, input_images=None, height=height, width=width,
					 text_guidance_scale=text_guidance, image_guidance_scale=1.0,
					 num_inference_steps=50, generator=gen, decouple_threshold=decouple_threshold)
			r.images[0].save(output_path)
		except Exception as e2:
			log(f"  [Step3] Fallback failed: {e2}")
	if not os.path.exists(output_path):
		log(f"  [Step3] Generation failed, missing output: {output_path}")
	return eff_img


# ==============================================================================
# Step 4: IPC Identity Diagnosis
# ==============================================================================
def ipc_diagnosis(prompt, image_path, vl_client, vl_model,
				  ref_image_path=None, domain="aircraft",
				  dino_extractor=None, f_log=None):
	log = lambda m: (f_log.write(m+"\n") if f_log else print(m))
	if not image_path or not os.path.exists(image_path):
		log(f"  [IPC] Skip diagnosis, image missing: {image_path}")
		return {
			"final_score": 0.0, "accepted": False,
			"identity_score": 0.0, "same_identity": False,
			"mismatch_cues": ["missing_generated_image"],
			"summary": "generated image missing",
			"dino_similarity": 0.0,
			"fix_strategy": "GLOBAL_REWRITE",
			"ipc_result": {"error": "generated image missing"},
		}
	move_helpers_to_gpu(retrieval_device)

	identity_result = {"same_identity":False,"identity_score":0.0,
					   "mismatch_cues":["no_reference"],"summary":"no reference","_parse_failed":True}
	if ref_image_path:
		try:
			identity_result = evaluate_identity_preservation(
				prompt=prompt, generated_image_path=image_path,
				reference_image_path=ref_image_path, client=vl_client, model=vl_model,
				domain=domain, generation_kwargs={"temperature":0.01,"max_tokens":512})
		except Exception as e:
			identity_result["summary"] = f"IPC error: {e}"
			log(f"  [IPC] Error: {e}")

	score       = float(identity_result.get("identity_score", 0.0) or 0.0)
	same_id     = bool(identity_result.get("same_identity", False))
	cues        = identity_result.get("mismatch_cues", [])
	if isinstance(cues, str): cues = [cues]
	cues = [str(x).strip() for x in (cues or []) if str(x).strip()]

	accepted = same_id and score >= 4.0
	final_score = 8.0 if accepted else score

	log(f"  [IPC] score={score:.2f}, same_id={same_id}, accepted={accepted}")
	if cues: log(f"  [IPC] cues: {cues[:3]}")

	dino_sim = 0.0
	if dino_extractor and ref_image_path:
		try:
			dino_sim = dino_extractor.compute_similarity(image_path, ref_image_path)
			log(f"  [IPC-DINO] sim={dino_sim:.4f}")
		except: pass

	if score < 4.0:   fix = "GLOBAL_REWRITE"
	elif score < 6.0: fix = "ENTITY_REFINE"
	elif score < 8.0: fix = "DETAIL_ENHANCE"
	else:             fix = "ACCEPT"

	return {
		"final_score": final_score, "accepted": accepted,
		"identity_score": score, "same_identity": same_id,
		"mismatch_cues": cues,
		"summary": str(identity_result.get("summary","") or ""),
		"dino_similarity": dino_sim,
		"fix_strategy": fix,
		"ipc_result": identity_result,
	}


# ==============================================================================
# Step 5a: LLM Prompt Reinforcement (IPC cues only)
# ==============================================================================
def llm_reinforce_prompt(gen_prompt, text_client, entity_name,
						  mismatch_cues=None, f_log=None):
	"""LLM 基于 IPC mismatch_cues 强化 generation_prompt。"""
	log = lambda m: (f_log.write(m+"\n") if f_log else print(m))
	if not mismatch_cues:
		return gen_prompt
	if text_client is None:
		return rule_reinforce_prompt(gen_prompt, mismatch_cues, entity_name, f_log=f_log)

	cues_str = "; ".join(f"'{c}'" for c in (mismatch_cues or [])[:4])
	msg = f"""You are a text-to-image prompt engineer.

Original prompt:
"{gen_prompt}"

Target entity: {entity_name}
Identity mismatch diagnostics from IPC: {cues_str or '(none)'}.

Rewrite the prompt to STRONGLY EMPHASIZE identity-corrective cues:
- Move critical identity attributes to the BEGINNING
- Use phrases like "clearly visible", "prominently featuring", "must show"
- Repeat key identity terms for emphasis
- Preserve overall realism and photographic coherence

Output ONLY the new prompt. No explanation, no quotes, no markdown."""

	try:
		new_prompt = call_text_api(text_client, [{"role": "user", "content": msg}],
									max_tokens=512).strip().strip('"').strip("'").strip()
		if not new_prompt or len(new_prompt) < 20:
			return gen_prompt
		log(f"  [LLMReinforce] New prompt[:200]: {new_prompt[:200]}")
		return new_prompt
	except Exception as e:
		log(f"  [LLMReinforce] Error: {e}, keep original.")
		return gen_prompt


def rule_reinforce_prompt(gen_prompt, mismatch_cues, entity_name, f_log=None):
	log = lambda m: (f_log.write(m+"\n") if f_log else print(m))
	clean_cues = [str(c).strip() for c in (mismatch_cues or []) if str(c).strip()]
	if not clean_cues:
		return gen_prompt
	prefix = f"a photo of a {entity_name}, clearly showing " if entity_name else "clearly showing "
	new_prompt = f"{prefix}{', '.join(clean_cues[:4])}. {gen_prompt}".strip()
	log(f"  [RuleReinforce] New prompt[:200]: {new_prompt[:200]}")
	return new_prompt


def resolve_retrieval_device():
	if args.retrieval_cpu:
		print("[Setup] retrieval_device -> cpu (--retrieval_cpu)")
		return "cpu"

	large_imagenet_retrieval = (
		"imagenet" in set(args.retrieval_datasets)
		and args.retrieval_method in {"CLIP", "LongCLIP", "SigLIP", "SigLIP2"}
	)
	single_gpu_mode = not args.vlm_device_id or args.vlm_device_id == args.device_id
	if large_imagenet_retrieval and single_gpu_mode:
		print(
			"[Setup] retrieval_device -> cpu (auto fallback for single-GPU large retrieval with ImageNet; "
			"prevents LongCLIP/SigLIP GPU OOM and missing output images)."
		)
		return "cpu"

	print("[Setup] retrieval_device -> cuda")
	return "cuda"


# ==============================================================================
# VRAM Management
# ==============================================================================
GLOBAL_QWEN_MODEL = None; GLOBAL_QWEN_PROCESSOR = None
retriever = None; dino_extractor_global = None; retrieval_device = "cuda"


def should_move_shared_qwen_with_retrieval():
	return args.retrieval_method in {"Qwen3-VL", "Qwen2.5-VL"}

def move_helpers_to_cpu():
	if args.vlm_device_id and args.vlm_device_id != args.device_id: return
	objects = [getattr(retriever,'model',None) if retriever else None, dino_extractor_global]
	if should_move_shared_qwen_with_retrieval():
		objects.insert(0, GLOBAL_QWEN_MODEL)
	for obj in objects:
		if obj:
			try: obj.to("cpu")
			except: pass
	gc.collect(); torch.cuda.empty_cache()

def move_helpers_to_gpu(target):
	if args.vlm_device_id and args.vlm_device_id != args.device_id: return
	gc.collect(); torch.cuda.empty_cache()
	objects = [getattr(retriever,'model',None) if retriever else None, dino_extractor_global]
	if should_move_shared_qwen_with_retrieval():
		objects.insert(0, GLOBAL_QWEN_MODEL)
	for obj in objects:
		if obj:
			try: obj.to(target)
			except: pass
	torch.cuda.empty_cache()


# ==============================================================================
# Dataset Loading
# ==============================================================================
def load_db():
	splits = {}
	for ds in args.retrieval_datasets:
		paths = []
		if ds == 'aircraft':
			root = "datasets/fgvc-aircraft-2013b/data/images"
			lf   = "datasets/fgvc-aircraft-2013b/data/images_train.txt"
			if os.path.exists(lf):
				paths = [os.path.join(root,l.strip()+".jpg") for l in open(lf) if l.strip()]
		elif ds == 'cub':
			root = "datasets/CUB_200_2011/images"
			sf = "datasets/CUB_200_2011/train_test_split.txt"
			im = "datasets/CUB_200_2011/images.txt"
			if os.path.exists(sf) and os.path.exists(im):
				tids = {l.split()[0] for l in open(sf) if len(l.split())>=2 and l.split()[1]=='1'}
				paths = [os.path.join(root,l.split()[1]) for l in open(im)
						 if len(l.split())>=2 and l.split()[0] in tids]
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
					paths = [os.path.join(imagenet_root, line.strip()) for line in f if line.strip()]
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
				print(f"  [WARNING] imagenet: no images found in common dirs. "
					  f"Retriever will use cached embeddings only if image_paths is non-empty.")
				emb_dir = f"datasets/embeddings/imagenet"
				import glob as _gl
				pts = _gl.glob(os.path.join(emb_dir, "longclip_embeddings_b*.pt"))
				if pts:
					dummy = torch.load(pts[0], map_location="cpu", weights_only=False)
					paths = dummy.get("paths", [])
					print(f"  imagenet: loaded {len(paths)} paths from embedding cache")
		splits[ds] = paths; print(f"  {ds}: {len(paths)}")
	return splits


# ==============================================================================
# Setup
# ==============================================================================
def setup_system(omnigen_device, vlm_device_map, shared_model=None, shared_proc=None):
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
		print(f"OmniGen2 load error: {e}"); sys.exit(1)

	if args.text_api_key:
		text_client = UsageTrackingClient(
			openai.OpenAI(api_key=args.text_api_key, base_url=args.text_api_base))
		print(f"[Setup] text_client -> {args.text_model} ({args.text_api_base})")
	else:
		text_client = None
		print("[Setup] text_client disabled: --text_api_key not provided, Step1/Step5-A use rule fallback.")

	if args.vl_api_key:
		vl_client = UsageTrackingClient(
			openai.OpenAI(api_key=args.vl_api_key, base_url=args.text_api_base))
		vl_model = args.vl_llm_model
	else:
		vl_client = UsageTrackingClient(LocalQwen3VLWrapper(
			args.local_model_weight_path, device_map=vlm_device_map,
			shared_model=shared_model, shared_processor=shared_proc))
		vl_model = "local-qwen3-vl"

	return pipe, text_client, vl_client, vl_model


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
	import numpy as np
	start_time = time.time()
	seed_everything(args.seed)

	dt       = datetime.now()
	out_root = f"results/{args.retrieval_method}/{dt.strftime('%Y.%-m.%-d')}/OmniGenV2_IPC_AR_{dt.strftime('%H-%M-%S')}"
	print(f"[Main] Output: {out_root}")

	retrieval_db = load_db()
	retrieval_device = resolve_retrieval_device()

	try:
		from transformers import AutoProcessor, AutoModelForImageTextToText
		GLOBAL_QWEN_PROCESSOR = AutoProcessor.from_pretrained(args.local_model_weight_path, trust_remote_code=True)
		GLOBAL_QWEN_MODEL = AutoModelForImageTextToText.from_pretrained(
			args.local_model_weight_path, torch_dtype=torch.bfloat16,
			device_map=vlm_device_map, trust_remote_code=True, low_cpu_mem_usage=True).eval()
	except Exception as e:
		print(f"[Main] VL load warning: {e}")

	class MultiDatasetRetriever:
		def __init__(self, splits, method, device, k, hybrid, ext_model, ext_proc):
			self.k = k
			is_qwen = method in ["Qwen3-VL","Qwen2.5-VL"]
			self.subs = [ImageRetriever(image_paths=paths,
				embeddings_path=f"datasets/embeddings/{ds}",
				method=method, device=device, k=k, use_hybrid=hybrid,
				external_model=ext_model if is_qwen else None,
				external_processor=ext_proc if is_qwen else None)
				for ds, paths in splits.items()]
		def search(self, queries):
			all_p, all_s = [], []
			for r in self.subs:
				p, s = r.search(queries); all_p.extend(p); all_s.extend(s)
			if not all_p: return [], []
			combined = sorted(zip(all_p,all_s),key=lambda x:x[1],reverse=True)[:self.k]
			return [x[0] for x in combined],[x[1] for x in combined]

	retriever = MultiDatasetRetriever(
		splits=retrieval_db, method=args.retrieval_method,
		device=retrieval_device, k=args.var_k,
		hybrid=args.use_hybrid_retrieval,
		ext_model=GLOBAL_QWEN_MODEL, ext_proc=GLOBAL_QWEN_PROCESSOR)
	torch.cuda.empty_cache()

	try:
		dino_extractor_global = DINOv3FeatureExtractor(
			args.dino_model_path, args.dino_weights_path, device=retrieval_device)
	except Exception as e:
		print(f"[Main] DINO init failed: {e}"); dino_extractor_global = None

	pipe, text_client, vl_client, vl_model = setup_system(
		omnigen_device, vlm_device_map,
		shared_model=GLOBAL_QWEN_MODEL, shared_proc=GLOBAL_QWEN_PROCESSOR)

	os.makedirs(out_root, exist_ok=True)
	logs_dir = os.path.join(out_root,"logs"); os.makedirs(logs_dir, exist_ok=True)
	run_metadata = collect_run_metadata(__file__, args, result_dir=out_root, logs_dir=logs_dir)
	write_run_metadata_json(logs_dir, run_metadata)

	monitor = ResourceMonitor(1.0); monitor.start()

	with open(os.path.join(logs_dir, "run_config.txt"), "w") as f:
		f.write("=== IPC-AR Pipeline Run Config ===\n")
		write_run_metadata_block(f, run_metadata)

	classes_txt = "datasets/fgvc-aircraft-2013b/data/variants.txt"
	with open(classes_txt) as f: all_classes = [l.strip() for l in f if l.strip()]
	my_tasks = [c for i,c in enumerate(all_classes) if i % args.total_chunks == args.task_index]
	if args.warmup_n_classes > 0: my_tasks = my_tasks[:args.warmup_n_classes]
	print(f"[Main] {len(my_tasks)}/{len(all_classes)} classes, max_retries={args.max_retries}")

	warmup = {"classes":0,"input_recovery":0,"ipc_calls":0,"accepts":0}

	for cls_idx, class_name in enumerate(tqdm(my_tasks, desc="IPC Pipeline")):
		safe  = class_name.replace(" ","_").replace("/","-")
		prompt= f"a photo of a {class_name}"
		f_log = open(os.path.join(out_root,f"{safe}.log"),"w")
		f_log.write(f"{'='*60}\nClass: {class_name}\n{'='*60}\n\n")
		write_run_metadata_block(f_log, run_metadata)

		try:
			f_log.write(">>> STEP 1: Input Interpreter (LLM API)\n")
			move_helpers_to_gpu(retrieval_device)
			interp     = importance_aware_input_interpreter(prompt, text_client, domain="aircraft")
			if args.warmup_n_classes > 0:
				warmup["classes"] += 1
				if interp.get("_parser_recovery_used"): warmup["input_recovery"] += 1

			gen_prompt = interp["generation_prompt"]
			ret_query  = interp["retrieval_query"]
			f_log.write(f"  Entity: {interp['high_importance']['entity']}\n")
			f_log.write(f"  GenPrompt[:150]: {gen_prompt[:150]}\n\n")

			f_log.write(">>> STEP 2: VAR Retrieval\n")
			valid_refs, best_score, var_details = dual_stage_retrieval(
				ret_query, retriever, vl_client, vl_model, var_k=args.var_k, f_log=f_log)
			best_ref = valid_refs[0] if valid_refs else None
			f_log.write(f"  best_ref: {os.path.basename(best_ref) if best_ref else 'None'}\n\n")

			f_log.write(">>> STEP 3: DINO-Injected Generation\n")
			v1_path    = os.path.join(out_root,f"{safe}_V1.png")
			cur_lambda = args.dino_lambda_init
			dino_injected_generation(pipe=pipe, generation_prompt=gen_prompt,
				ref_image_path=best_ref, dino_extractor=dino_extractor_global,
				dino_lambda=cur_lambda, seed=args.seed, output_path=v1_path, f_log=f_log,
				height=args.height, width=args.width,
				img_guidance=args.image_guidance_scale, text_guidance=args.text_guidance_scale,
				decouple_threshold=args.decouple_threshold)
			if not os.path.exists(v1_path):
				f_log.write("  [Step3] Initial generation failed; skip class.\n")
				continue

			f_log.write(">>> STEP 4: IPC Identity Diagnosis\n")
			best_image = v1_path; best_final_score = -1; retry_cnt = 0

			diag = ipc_diagnosis(prompt, v1_path, vl_client, vl_model,
				ref_image_path=best_ref, domain="aircraft",
				dino_extractor=dino_extractor_global, f_log=f_log)

			if args.warmup_n_classes > 0:
				warmup["ipc_calls"] += 1
				if diag["accepted"]: warmup["accepts"] += 1
			accepted = diag["accepted"]; best_final_score = diag["final_score"]
			best_scored_image = v1_path; best_scored_value = diag["final_score"]
			current_gen_prompt = gen_prompt
			if accepted: best_image = v1_path

			if not accepted and args.max_retries >= 1:
				f_log.write(f">>> STEP 5: Retry (max={args.max_retries}, TIFA=OFF)\n")
				while retry_cnt < args.max_retries:
					retry_cnt += 1
					next_path = os.path.join(out_root, f"{safe}_V{retry_cnt+1}.png")
					cur_ref   = (valid_refs[retry_cnt]
								 if valid_refs and retry_cnt < len(valid_refs) else best_ref)

					if diag.get("mismatch_cues"):
						f_log.write(f"  [Step5-A] LLM reinforce; IPC_cues={diag.get('mismatch_cues',[])[:3]}\n")
						current_gen_prompt = llm_reinforce_prompt(
							current_gen_prompt, text_client,
							entity_name=interp["high_importance"]["entity"],
							mismatch_cues=diag.get("mismatch_cues", []),
							f_log=f_log)

					cur_lambda = min(cur_lambda + args.dino_lambda_step, args.dino_lambda_max)
					dino_injected_generation(pipe=pipe, generation_prompt=current_gen_prompt,
						ref_image_path=cur_ref, dino_extractor=dino_extractor_global,
						dino_lambda=cur_lambda, seed=args.seed+retry_cnt, output_path=next_path,
						f_log=f_log, height=args.height, width=args.width,
						img_guidance=args.image_guidance_scale, text_guidance=args.text_guidance_scale,
						decouple_threshold=args.decouple_threshold)
					if not os.path.exists(next_path):
						f_log.write(f"  [Step5] retry={retry_cnt} generation failed, skip diagnosis.\n")
						if retry_cnt >= args.max_retries:
							best_image = best_scored_image; best_final_score = best_scored_value
							break
						continue

					diag = ipc_diagnosis(prompt, next_path, vl_client, vl_model,
						ref_image_path=best_ref, domain="aircraft",
						dino_extractor=dino_extractor_global, f_log=f_log)
					if args.warmup_n_classes > 0:
						warmup["ipc_calls"] += 1
						if diag["accepted"]: warmup["accepts"] += 1
					if diag["final_score"] > best_scored_value:
						best_scored_image = next_path
						best_scored_value = diag["final_score"]
					if diag["accepted"]:
						best_image = next_path; best_final_score = diag["final_score"]
						f_log.write(f"  [Step5] ACCEPT retry={retry_cnt}\n"); break
					if retry_cnt >= args.max_retries:
						best_image = best_scored_image; best_final_score = best_scored_value
						f_log.write(
							f"  [Step5] retry={retry_cnt} select best-scored image: "
							f"{os.path.basename(best_image)} score={best_final_score:.2f}\n")
						break
					best_image = best_scored_image

			final_path = os.path.join(out_root,f"{safe}_FINAL.png")
			if best_image and os.path.exists(best_image): shutil.copy(best_image, final_path)
			f_log.write(f"\n>>> FINAL: {os.path.basename(best_image)} score={best_final_score}\n")

			json.dump({
				"class_name":class_name,"prompt":prompt,
				"script_name": run_metadata["script_name"],
				"script_path": run_metadata["script_path"],
				"command_redacted": run_metadata["command_redacted"],
				"entity":interp["high_importance"]["entity"],
				"generation_prompt":gen_prompt[:500],
				"best_ref":os.path.basename(best_ref) if best_ref else None,
				"final_score":best_final_score,"total_retries":retry_cnt,
				"dino_lambda_final":cur_lambda,
				"ipc_accepted":"yes" if best_final_score>=6.0 else "no",
			}, open(os.path.join(logs_dir,f"{safe}_summary.json"),"w"), indent=2, ensure_ascii=False)

		except Exception as e:
			import traceback
			f_log.write(f"\n>>> EXCEPTION: {e}\n{traceback.format_exc()}")
		finally:
			f_log.close()

	monitor.stop(); monitor.save_plots(logs_dir)
	elapsed = time.time()-start_time
	with open(os.path.join(logs_dir,"pipeline_summary.txt"),"w") as f:
		f.write(f"IPC Pipeline | {elapsed:.1f}s | {len(my_tasks)} classes\n")
		write_run_metadata_block(f, run_metadata)
		f.write(f"text_model={args.text_model} | vl={vl_model}\n")
		f.write(f"Tokens: in={RUN_STATS['input_tokens']}, out={RUN_STATS['output_tokens']}\n")
		if args.warmup_n_classes > 0:
			bc = max(1,warmup["ipc_calls"]); cc = max(1,warmup["classes"])
			f.write(f"Input recovery: {warmup['input_recovery']}/{cc}\n")
			f.write(f"IPC accept rate: {warmup['accepts']}/{bc} ({warmup['accepts']/bc:.2%})\n")
	print(f"\n[Done] {elapsed:.1f}s | {out_root}")
