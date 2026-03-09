import itertools
import json
import os
import re
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.critical.taxonomy_aware_critic import (
    _extract_json_candidate,
    _normalize_tac_response,
    _parse_json_loose,
    message_gpt,
)


def _tac_schema_errors(parsed):
    errors = []
    if not isinstance(parsed, dict):
        return ["not_dict"]
    required = {
        "status": str,
        "final_score": (int, float),
        "taxonomy_check": str,
        "critique": str,
        "refined_prompt": str,
        "retrieval_queries": list,
        "error_analysis": dict,
        "refined_negative_prompt": str,
    }
    for key, expected_type in required.items():
        if key not in parsed:
            errors.append(f"missing_key:{key}")
            continue
        if not isinstance(parsed[key], expected_type):
            errors.append(f"wrong_type:{key}:{type(parsed[key]).__name__}")

    if isinstance(parsed.get("taxonomy_check"), str):
        if parsed["taxonomy_check"] not in {"correct", "wrong_subtype", "wrong_object", "error"}:
            errors.append(f"invalid_taxonomy:{parsed['taxonomy_check']}")
    if isinstance(parsed.get("retrieval_queries"), list):
        if not all(isinstance(x, str) for x in parsed["retrieval_queries"]):
            errors.append("retrieval_queries_not_all_str")
    return errors


def _input_schema_errors(parsed):
    errors = []
    if not isinstance(parsed, dict):
        return ["not_dict"]

    if not isinstance(parsed.get("high_importance"), dict):
        errors.append("high_importance_not_dict")
    else:
        hi = parsed["high_importance"]
        if not isinstance(hi.get("entity"), str):
            errors.append("high_importance.entity_not_str")
        if not isinstance(hi.get("key_features"), list):
            errors.append("high_importance.key_features_not_list")

    if not isinstance(parsed.get("low_importance"), dict):
        errors.append("low_importance_not_dict")
    else:
        lo = parsed["low_importance"]
        for key in ["background", "composition", "lighting", "visual_style"]:
            if not isinstance(lo.get(key), str):
                errors.append(f"low_importance.{key}_not_str")

    if not isinstance(parsed.get("importance_weights"), dict):
        errors.append("importance_weights_not_dict")
    else:
        iw = parsed["importance_weights"]
        for key in ["entity_identity", "structural_detail", "environment", "artistic_style"]:
            if not isinstance(iw.get(key), (int, float)):
                errors.append(f"importance_weights.{key}_not_number")

    if not isinstance(parsed.get("retrieval_query"), str):
        errors.append("retrieval_query_not_str")
    if not isinstance(parsed.get("generation_prompt"), str):
        errors.append("generation_prompt_not_str")
    if not isinstance(parsed.get("ambiguous_elements"), list):
        errors.append("ambiguous_elements_not_list")

    return errors


def _canon_key(key):
    if not isinstance(key, str):
        return key
    k = key.strip().lower().replace("-", "_").replace(" ", "_")
    k = re.sub(r"_+", "_", k)
    while k.startswith("_"):
        k = k[1:]
    return k


def _normalize_keys(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[_canon_key(k)] = _normalize_keys(v)
        return out
    if isinstance(obj, list):
        return [_normalize_keys(x) for x in obj]
    return obj


def _normalize_input_output(parsed):
    if not isinstance(parsed, dict):
        return parsed

    data = _normalize_keys(parsed)
    alias = {
        "highimportance": "high_importance",
        "lowimportance": "low_importance",
        "importanceweights": "importance_weights",
        "retrievalquery": "retrieval_query",
        "retrival_query": "retrieval_query",
        "retrieval_queries": "retrieval_query",
        "generaion_prompt": "generation_prompt",
        "ambiguouselements": "ambiguous_elements",
    }
    for src, dst in alias.items():
        if src in data and dst not in data:
            data[dst] = data[src]

    data.setdefault("high_importance", {})
    data.setdefault("low_importance", {})
    data.setdefault("importance_weights", {})
    data.setdefault("retrieval_query", "")
    data.setdefault("generation_prompt", "")
    data.setdefault("ambiguous_elements", [])

    if not isinstance(data["high_importance"], dict):
        data["high_importance"] = {}
    data["high_importance"].setdefault("entity", "")
    data["high_importance"].setdefault("key_features", [])
    if isinstance(data["high_importance"].get("key_features"), str):
        data["high_importance"]["key_features"] = [data["high_importance"]["key_features"]]

    if not isinstance(data["low_importance"], dict):
        data["low_importance"] = {}
    for key in ["background", "composition", "lighting", "visual_style"]:
        data["low_importance"].setdefault(key, "")

    if not isinstance(data["importance_weights"], dict):
        data["importance_weights"] = {}
    for key in ["entity_identity", "structural_detail", "environment", "artistic_style"]:
        val = data["importance_weights"].get(key, 0.0)
        if not isinstance(val, (int, float)):
            try:
                val = float(val)
            except Exception:
                val = 0.0
        data["importance_weights"][key] = float(val)

    if not isinstance(data.get("retrieval_query"), str):
        data["retrieval_query"] = str(data.get("retrieval_query", ""))
    if not isinstance(data.get("generation_prompt"), str):
        data["generation_prompt"] = str(data.get("generation_prompt", ""))
    if not isinstance(data.get("ambiguous_elements"), list):
        data["ambiguous_elements"] = []
    data["ambiguous_elements"] = [str(x) for x in data["ambiguous_elements"]]

    return data


def _normalize_tac_output(parsed):
    if not isinstance(parsed, dict):
        return parsed

    data = _normalize_keys(parsed)
    alias = {
        "finalscore": "final_score",
        "taxonomycheck": "taxonomy_check",
        "critiques": "critique",
        "critique_": "critique",
        "critiques_": "critique",
        "retrievalquery": "retrieval_queries",
        "retrievalqueries": "retrieval_queries",
        "retrieved_negative_prompt": "refined_negative_prompt",
        "refined_negative_prompts": "refined_negative_prompt",
        "erroranalysis": "error_analysis",
    }
    for src, dst in alias.items():
        if src in data and dst not in data:
            data[dst] = data[src]

    data.setdefault("status", "error")
    data.setdefault("final_score", 0)
    data.setdefault("taxonomy_check", "error")
    data.setdefault("critique", "")
    data.setdefault("refined_prompt", "")
    data.setdefault("retrieval_queries", [])
    data.setdefault("error_analysis", {"type": "Global"})
    data.setdefault("refined_negative_prompt", "")

    if not isinstance(data["final_score"], (int, float)):
        try:
            data["final_score"] = float(data["final_score"])
        except Exception:
            data["final_score"] = 0.0

    if not isinstance(data["retrieval_queries"], list):
        if isinstance(data["retrieval_queries"], str):
            data["retrieval_queries"] = [data["retrieval_queries"]]
        else:
            data["retrieval_queries"] = []
    data["retrieval_queries"] = [str(x) for x in data["retrieval_queries"]]

    if not isinstance(data["error_analysis"], dict):
        data["error_analysis"] = {"type": "Global"}
    data["error_analysis"].setdefault("type", "Global")

    return data


def _input_prompt(user_prompt):
    return f"""
You are an Input Interpreter Agent for aircraft generation.
User prompt: \"{user_prompt}\".

Return EXACTLY one JSON object with these keys and types:
- high_importance: object {{entity: string, key_features: string[]}}
- low_importance: object {{background: string, composition: string, lighting: string, visual_style: string}}
- importance_weights: object {{entity_identity: number, structural_detail: number, environment: number, artistic_style: number}}
- retrieval_query: string (format: \"a photo of a [entity]\")
- generation_prompt: string
- ambiguous_elements: string[]

No markdown, no prose, no extra keys.
If uncertain, still output valid defaults using the same schema.
""".strip()


def _input_prompt_template_fill(user_prompt):
        return f"""
Fill values in the JSON template below. Keep all keys exactly unchanged and keep valid JSON syntax.
User prompt: \"{user_prompt}\".

{{
    "high_importance": {{"entity": "", "key_features": []}},
    "low_importance": {{"background": "", "composition": "", "lighting": "", "visual_style": ""}},
    "importance_weights": {{"entity_identity": 0.0, "structural_detail": 0.0, "environment": 0.0, "artistic_style": 0.0}},
    "retrieval_query": "",
    "generation_prompt": "",
    "ambiguous_elements": []
}}

Output only JSON.
""".strip()


def _tac_prompt(user_prompt):
    return f"""
Analyze the image for prompt \"{user_prompt}\".
Return EXACTLY one JSON object with keys:
status, final_score, taxonomy_check, critique, refined_prompt, retrieval_queries, error_analysis, refined_negative_prompt.

Type constraints:
- status: \"success\" or \"error\"
- final_score: number in [0, 10]
- taxonomy_check: one of [\"correct\",\"wrong_subtype\",\"wrong_object\",\"error\"]
- critique: string
- refined_prompt: string
- retrieval_queries: string[]
- error_analysis: object {{type: \"Global\" or \"Local\"}}
- refined_negative_prompt: string

No markdown, no prose, no extra keys.
If uncertain, still output valid defaults with this exact schema.
""".strip()


def _tac_prompt_template_fill(user_prompt):
        return f"""
Analyze the image for prompt \"{user_prompt}\" and fill this JSON template.
Keep all keys exactly unchanged and keep valid JSON syntax.

{{
    "status": "success",
    "final_score": 0.0,
    "taxonomy_check": "correct",
    "critique": "",
    "refined_prompt": "",
    "retrieval_queries": [],
    "error_analysis": {{"type": "Global"}},
    "refined_negative_prompt": ""
}}

Output only JSON.
""".strip()


def _repair_prompt(task_name, schema_errors, previous_output, original_prompt):
    err_text = ", ".join(schema_errors[:12]) if schema_errors else "unknown_schema_error"
    return (
        f"Your previous {task_name} JSON output is invalid.\n"
        f"Schema errors: {err_text}\n"
        f"Previous output: {previous_output[:1200]}\n\n"
        f"Now output ONLY valid JSON that satisfies this instruction:\n{original_prompt}\n"
    )


def _call_with_retry(
    client,
    model_name,
    base_prompt,
    image_paths,
    gen_kwargs,
    validator,
    normalizer,
    retries=2,
):
    prompt = base_prompt
    attempts = []

    for idx in range(retries + 1):
        raw = message_gpt(
            msg=prompt,
            client=client,
            image_paths=image_paths,
            model=model_name,
            **gen_kwargs,
        )
        candidate = _extract_json_candidate(raw)
        parsed = _parse_json_loose(candidate)
        strict_errors = validator(parsed)
        normalized = normalizer(parsed) if isinstance(parsed, dict) else None
        normalized_errors = validator(normalized)

        attempts.append(
            {
                "attempt": idx + 1,
                "raw_preview": (raw or "")[:1200],
                "candidate_preview": (candidate or "")[:1200],
                "parse_ok": isinstance(parsed, dict),
                "strict_errors": strict_errors,
                "normalized_errors": normalized_errors,
            }
        )

        if isinstance(normalized, dict) and not normalized_errors:
            return {
                "success": True,
                "parsed": parsed,
                "normalized": normalized,
                "attempts": attempts,
            }

        if idx < retries:
            prompt = _repair_prompt(
                task_name="QwenVL",
                schema_errors=normalized_errors,
                previous_output=(raw or ""),
                original_prompt=base_prompt,
            )

    return {
        "success": False,
        "parsed": parsed if isinstance(parsed, dict) else None,
        "normalized": normalized if isinstance(normalized, dict) else None,
        "attempts": attempts,
    }


class TestQwenVLAutotune(unittest.TestCase):
    def test_autotune_prompt_params_and_flow(self):
        if os.getenv("RUN_QWEN_AUTOTUNE", "0") != "1":
            self.skipTest("Set RUN_QWEN_AUTOTUNE=1 to run QwenVL autotune")

        from src.utils.rag_utils import LocalQwen3VLWrapper

        model_path = os.getenv("QWEN_LOCAL_MODEL_PATH", "/home/tingyu/imageRAG/Qwen3-VL-4B-Instruct")
        image_path = os.getenv(
            "QWEN_PROBE_IMAGE",
            "/home/tingyu/imageRAG/results/LongCLIP/2026.3.8/OmniGenV2_DualPath_AR_20-47-54/707-320_V1.png",
        )
        model_name = os.getenv("QWEN_MODEL_NAME", "local-qwen3-vl")
        device_map = os.getenv("QWEN_DEVICE_MAP", "auto")
        max_cases = int(os.getenv("QWEN_AUTOTUNE_MAX_CASES", "12"))

        self.assertTrue(os.path.exists(model_path), f"Model path not found: {model_path}")
        self.assertTrue(os.path.exists(image_path), f"Probe image not found: {image_path}")

        client = LocalQwen3VLWrapper(model_path=model_path, device_map=device_map)

        user_prompt = "a photo of a 707-320"
        prompt_variants = [
            {
                "name": "strict_contract",
                "input_prompt": _input_prompt(user_prompt),
                "tac_prompt": _tac_prompt(user_prompt),
            },
            {
                "name": "template_fill",
                "input_prompt": _input_prompt_template_fill(user_prompt),
                "tac_prompt": _tac_prompt_template_fill(user_prompt),
            },
        ]

        param_grid = [
            {
                "name": "sample_035",
                "temperature": 0.35,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.0,
                "max_new_tokens": 512,
                "do_sample": True,
            },
            {
                "name": "sample_020",
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.0,
                "max_new_tokens": 512,
                "do_sample": True,
            },
            {
                "name": "greedy",
                "temperature": 0,
                "repetition_penalty": 1.0,
                "max_new_tokens": 512,
                "do_sample": False,
            },
            {
                "name": "short_seq",
                "temperature": 0.35,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.0,
                "max_new_tokens": 256,
                "do_sample": True,
            },
        ]

        flow_grid = [
            {"name": "retry1", "retries": 1},
            {"name": "retry2", "retries": 2},
            {"name": "retry3", "retries": 3},
        ]

        all_cases = list(itertools.product(param_grid, flow_grid, prompt_variants))[:max_cases]
        results = []

        for param_cfg, flow_cfg, prompt_cfg in all_cases:
            text_kwargs = dict(param_cfg)
            vl_kwargs = dict(param_cfg)

            input_res = _call_with_retry(
                client=client,
                model_name=model_name,
                base_prompt=prompt_cfg["input_prompt"],
                image_paths=[],
                gen_kwargs=text_kwargs,
                validator=_input_schema_errors,
                normalizer=_normalize_input_output,
                retries=flow_cfg["retries"],
            )

            tac_res = _call_with_retry(
                client=client,
                model_name=model_name,
                base_prompt=prompt_cfg["tac_prompt"],
                image_paths=[image_path],
                gen_kwargs=vl_kwargs,
                validator=_tac_schema_errors,
                normalizer=_normalize_tac_output,
                retries=flow_cfg["retries"],
            )

            normalized_tac = _normalize_tac_response(
                tac_res["parsed"] if isinstance(tac_res.get("parsed"), dict) else {},
                prompt=user_prompt,
            )

            task_success = int(input_res["success"]) + int(tac_res["success"])
            retries_used = (len(input_res["attempts"]) - 1) + (len(tac_res["attempts"]) - 1)
            score = task_success * 100 - retries_used * 5

            results.append(
                {
                    "param": param_cfg,
                    "flow": flow_cfg,
                    "prompt": prompt_cfg["name"],
                    "input_success": input_res["success"],
                    "tac_success": tac_res["success"],
                    "input_attempts": len(input_res["attempts"]),
                    "tac_attempts": len(tac_res["attempts"]),
                    "score": score,
                    "normalized_tac": {
                        "status": normalized_tac.get("status"),
                        "final_score": normalized_tac.get("final_score"),
                        "taxonomy_check": normalized_tac.get("taxonomy_check"),
                        "schema_warnings": normalized_tac.get("schema_warnings", []),
                    },
                    "input_last": input_res["attempts"][-1],
                    "tac_last": tac_res["attempts"][-1],
                }
            )

        best = sorted(results, key=lambda x: x["score"], reverse=True)[0]

        report = {
            "model_path": model_path,
            "image_path": image_path,
            "user_prompt": user_prompt,
            "search_space": {
                "param_grid": sorted(list({x[0]["name"] for x in all_cases})),
                "flow_grid": sorted(list({x[1]["name"] for x in all_cases})),
                "prompt_grid": sorted(list({x[2]["name"] for x in all_cases})),
            },
            "results": results,
            "best": best,
            "recommended_flow": [
                "1) 发送严格 JSON 提示词",
                "2) 解析并进行 schema 校验",
                "3) 若失败则带错误清单重试",
                "4) TAC 输出再做 normalize 兜底",
            ],
        }

        out_dir = os.path.join(PROJECT_ROOT, "temp", "qwenvl_probe")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "autotune_best_report.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.assertTrue(
            best["input_success"] and best["tac_success"],
            msg=f"No fully successful prompt+param+flow found. See {out_path}",
        )


if __name__ == "__main__":
    unittest.main()
