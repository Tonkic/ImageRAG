import json
import os
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


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse(self.content)


class _FakeChat:
    def __init__(self, content):
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content):
        self.chat = _FakeChat(content)


class TestQwenVLProbe(unittest.TestCase):
    @staticmethod
    def _schema_errors(data):
        schema_errors = []
        if isinstance(data, dict):
            required_schema = {
                "status": str,
                "final_score": (int, float),
                "taxonomy_check": str,
                "critique": str,
                "refined_prompt": str,
                "retrieval_queries": list,
                "error_analysis": dict,
                "refined_negative_prompt": str,
            }
            for key, expected_type in required_schema.items():
                if key not in data:
                    schema_errors.append(f"missing_key:{key}")
                    continue
                if not isinstance(data[key], expected_type):
                    schema_errors.append(
                        f"wrong_type:{key}:expected={expected_type}:got={type(data[key]).__name__}"
                    )
            if isinstance(data.get("taxonomy_check"), str):
                if data["taxonomy_check"] not in {"correct", "wrong_subtype", "wrong_object", "error"}:
                    schema_errors.append(f"invalid_value:taxonomy_check:{data['taxonomy_check']}")
            if isinstance(data.get("retrieval_queries"), list):
                if not all(isinstance(x, str) for x in data["retrieval_queries"]):
                    schema_errors.append("wrong_type:retrieval_queries_items:not_all_str")
        return schema_errors

    def test_message_gpt_forwards_generation_kwargs(self):
        client = _FakeClient('{"status":"ok"}')

        _ = message_gpt(
            msg="return json",
            client=client,
            image_paths=[],
            model="local-qwen3-vl",
            temperature=0.01,
            top_p=0.82,
            top_k=24,
            repetition_penalty=1.12,
            no_repeat_ngram_size=5,
            max_new_tokens=300,
        )

        sent = client.chat.completions.last_kwargs
        self.assertIsNotNone(sent)
        self.assertEqual(sent.get("temperature"), 0.01)
        self.assertEqual(sent.get("top_p"), 0.82)
        self.assertEqual(sent.get("top_k"), 24)
        self.assertEqual(sent.get("repetition_penalty"), 1.12)
        self.assertEqual(sent.get("no_repeat_ngram_size"), 5)
        self.assertEqual(sent.get("max_new_tokens"), 300)

    def test_tac_normalizer_repairs_schema_drift(self):
        parsed_like_qwen = {
            "status": "valid",
            "final_score": 0.95,
            "taxonomy_check": True,
            "critique": "",
            "refined_prompt": "",
            "retrieval_queries": ["Airbus A320 at airport"],
            "error_analysis": [],
            "refINED_negative_prompt": "",
        }

        normalized = _normalize_tac_response(parsed_like_qwen, prompt="a photo of a A320")
        errs = self._schema_errors(normalized)

        self.assertFalse(errs, msg=f"normalized schema invalid: {errs}")
        self.assertEqual(normalized["status"], "success")
        self.assertEqual(normalized["taxonomy_check"], "correct")
        self.assertAlmostEqual(normalized["final_score"], 9.5, places=3)
        self.assertIn("score_rescaled_0_1_to_0_10", normalized.get("schema_warnings", []))

    def test_qwenvl_raw_output_probe(self):
        if os.getenv("RUN_QWEN_VL_E2E", "0") != "1":
            self.skipTest("Set RUN_QWEN_VL_E2E=1 to run real QwenVL probe")

        from src.utils.rag_utils import LocalQwen3VLWrapper

        model_path = os.getenv("QWEN_LOCAL_MODEL_PATH", "/home/tingyu/imageRAG/Qwen3-VL-4B-Instruct")
        image_path = os.getenv(
            "QWEN_PROBE_IMAGE",
            "/home/tingyu/imageRAG/results/LongCLIP/2026.3.8/OmniGenV2_Ablation_noDINO_AR_01-10-59/A320_V1.png",
        )
        device_map = os.getenv("QWEN_DEVICE_MAP", "auto")

        if not os.path.exists(model_path):
            self.skipTest(f"Model path not found: {model_path}")
        if not os.path.exists(image_path):
            self.skipTest(f"Probe image not found: {image_path}")

        temperature = float(os.getenv("QWEN_PROBE_TEMPERATURE", "0.01"))
        top_p = float(os.getenv("QWEN_PROBE_TOP_P", "0.8"))
        top_k = int(os.getenv("QWEN_PROBE_TOP_K", "20"))
        repetition_penalty = float(os.getenv("QWEN_PROBE_REPETITION_PENALTY", "1.15"))
        no_repeat_ngram_size = int(os.getenv("QWEN_PROBE_NO_REPEAT_NGRAM", "4"))
        max_new_tokens = int(os.getenv("QWEN_PROBE_MAX_NEW_TOKENS", "512"))

        prompt = (
            "You are a strict JSON generator. Analyze this image and output ONLY valid JSON with keys: "
            "status, final_score, taxonomy_check, critique, refined_prompt, retrieval_queries, "
            "error_analysis, refined_negative_prompt. Do not output markdown or prose."
        )

        client = LocalQwen3VLWrapper(model_path=model_path, device_map=device_map)
        raw = message_gpt(
            msg=prompt,
            client=client,
            image_paths=[image_path],
            model="local-qwen3-vl",
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )

        candidate = _extract_json_candidate(raw)
        parsed = _parse_json_loose(candidate)

        raw_schema_errors = self._schema_errors(parsed)
        normalized = _normalize_tac_response(parsed if isinstance(parsed, dict) else {}, prompt="a photo of a A320")
        normalized_schema_errors = self._schema_errors(normalized)

        if not raw:
            failure_type = "empty_output"
        elif "{" not in (candidate or ""):
            failure_type = "no_json_object"
        elif parsed is None:
            failure_type = "json_parse_failed"
        elif raw_schema_errors:
            failure_type = "raw_schema_invalid"
        elif normalized_schema_errors:
            failure_type = "normalized_schema_invalid"
        else:
            failure_type = "none"

        report = {
            "model_path": model_path,
            "image_path": image_path,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "max_new_tokens": max_new_tokens,
            "raw_preview": (raw or "")[:2000],
            "candidate_preview": (candidate or "")[:2000],
            "failure_type": failure_type,
            "parsed_is_dict": isinstance(parsed, dict),
            "parsed_keys": sorted(list(parsed.keys())) if isinstance(parsed, dict) else [],
            "raw_schema_errors": raw_schema_errors,
            "normalized_schema_errors": normalized_schema_errors,
            "normalized_preview": {
                "status": normalized.get("status"),
                "final_score": normalized.get("final_score"),
                "taxonomy_check": normalized.get("taxonomy_check"),
                "retrieval_queries": normalized.get("retrieval_queries"),
                "schema_warnings": normalized.get("schema_warnings", []),
            },
        }

        report_dir = os.path.join(PROJECT_ROOT, "temp", "qwenvl_probe")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "latest_probe_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.assertTrue(raw and raw.strip(), msg=f"QwenVL returned empty output. See {report_path}")
        self.assertIsNotNone(parsed, msg=f"QwenVL output is not parseable JSON. See {report_path}")
        self.assertFalse(normalized_schema_errors, msg=f"TAC-normalized schema invalid. See {report_path}")


if __name__ == "__main__":
    unittest.main()
