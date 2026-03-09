import os
import sys
import unittest
from datetime import datetime
import re
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.critical.taxonomy_aware_critic import (
    _extract_json_candidate,
    _normalize_tac_contract_keys,
    _normalize_tac_response,
    _parse_json_loose,
    _validate_tac_schema,
    message_gpt,
)


class TestTacSchemaTriggerExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.getenv("RUN_QWEN_REAL_SCHEMA_TEST", "1") != "1":
            raise unittest.SkipTest("Set RUN_QWEN_REAL_SCHEMA_TEST=1 to run real Qwen schema tests")

        from src.utils.rag_utils import LocalQwen3VLWrapper

        model_path = os.getenv("QWEN_LOCAL_MODEL_PATH", "/home/tingyu/imageRAG/Qwen3-VL-4B-Instruct")
        model_name = os.getenv("QWEN_MODEL_NAME", "local-qwen3-vl")
        device_candidates = [x.strip() for x in os.getenv("QWEN_TEST_DEVICE_IDS", "1,2").split(",") if x.strip()]

        if not os.path.exists(model_path):
            raise unittest.SkipTest(f"Local model path not found: {model_path}")

        client = None
        last_err = None
        for dev in device_candidates:
            try:
                device_map = {"": f"cuda:{int(dev)}"}
                client = LocalQwen3VLWrapper(model_path=model_path, device_map=device_map)
                cls.device_id = int(dev)
                break
            except Exception as exc:
                last_err = exc

        if client is None:
            raise unittest.SkipTest(f"Unable to load Qwen on device 1/2. Last error: {last_err}")

        cls.client = client
        cls.model_name = model_name
        cls.report_dir = os.path.join(PROJECT_ROOT, "temp", "qwenvl_probe")
        os.makedirs(cls.report_dir, exist_ok=True)
        cls.records = []

        cls.run_dir = os.getenv(
            "QWEN_LOG_DRIVEN_RUN_DIR",
            "/home/tingyu/imageRAG/results/LongCLIP/2026.3.9/OmniGenV2_DualPath_AR_01-05-46",
        )
        cls.case_name = os.getenv("QWEN_LOG_DRIVEN_CASE", "737-300")
        cls.log_path = os.path.join(cls.run_dir, f"{cls.case_name}.log")

        if not os.path.exists(cls.log_path):
            raise unittest.SkipTest(f"Log file not found: {cls.log_path}")

        with open(cls.log_path, "r", encoding="utf-8", errors="ignore") as f:
            cls.log_text = f.read()

        prompt_match = re.search(r"^Prompt:\s*(.+)$", cls.log_text, flags=re.MULTILINE)
        if not prompt_match:
            raise unittest.SkipTest(f"Prompt not found in log: {cls.log_path}")
        cls.case_prompt = prompt_match.group(1).strip()

        cls.bad_image_path = os.path.join(cls.run_dir, f"{cls.case_name}_V1.png")
        final_match = re.search(r">>> FINAL:\s*(\S+)\s*→\s*FINAL", cls.log_text)
        if final_match:
            cls.good_image_path = os.path.join(cls.run_dir, final_match.group(1))
        else:
            cls.good_image_path = os.path.join(cls.run_dir, f"{cls.case_name}_V3.png")

        if not os.path.exists(cls.bad_image_path):
            raise unittest.SkipTest(f"Bad sample image not found: {cls.bad_image_path}")
        if not os.path.exists(cls.good_image_path):
            raise unittest.SkipTest(f"Good sample image not found: {cls.good_image_path}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "records") and cls.records:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(cls.report_dir, f"schema_trigger_realqwen_{ts}.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump({
                    "device_id": getattr(cls, "device_id", None),
                    "run_dir": getattr(cls, "run_dir", None),
                    "case_name": getattr(cls, "case_name", None),
                    "records": cls.records,
                }, f, ensure_ascii=False, indent=2)

    def _run_case(self, image_path, case_tag):
        prompt = (
            f"Analyze this image for prompt '{self.case_prompt}'. "
            "Return ONLY one JSON object with EXACTLY 3 keys: "
            "final_score (number in [0,10]), retrieval_queries (array of strings), refined_prompt (string). "
            "No markdown, no prose, no extra keys."
        )

        raw = message_gpt(
            msg=prompt,
            client=self.client,
            image_paths=[image_path],
            model=self.model_name,
            temperature=0.35,
            top_p=0.8,
            top_k=20,
            max_new_tokens=384,
            do_sample=True,
        )
        candidate = _extract_json_candidate(raw)
        parsed = _parse_json_loose(candidate)

        normalized_keys = _normalize_tac_contract_keys(parsed) if isinstance(parsed, dict) else {}
        errors = _validate_tac_schema(normalized_keys, compact_output=True)
        normalized = _normalize_tac_response(normalized_keys, prompt=self.case_prompt)

        self.records.append(
            {
                "case_tag": case_tag,
                "case_prompt": self.case_prompt,
                "image_path": image_path,
                "raw_preview": (raw or "")[:1200],
                "parse_ok": isinstance(parsed, dict),
                "schema_errors": errors,
                "schema_warnings": normalized.get("schema_warnings", []),
                "normalized_final_score": normalized.get("final_score"),
                "normalized_refined_prompt": normalized.get("refined_prompt", "")[:200],
                "normalized_queries": normalized.get("retrieval_queries", []),
            }
        )
        return raw, parsed, errors, normalized

    def test_real_qwenvl_log_driven_good_and_bad_inputs(self):
        bad_raw, bad_parsed, bad_errors, bad_norm = self._run_case(self.bad_image_path, "bad_from_log_v1")
        good_raw, good_parsed, good_errors, good_norm = self._run_case(self.good_image_path, "good_from_log_final")

        self.assertTrue(isinstance(bad_raw, str) and len(bad_raw.strip()) > 0)
        self.assertTrue(isinstance(good_raw, str) and len(good_raw.strip()) > 0)

        # 该测试目标是“观测真实运行输入下 schema 行为”，允许 bad/good 任一出现 schema 问题。
        # 只要求至少有一侧能被解析成 dict，避免把测试变成脆弱的质量评测。
        self.assertTrue(
            isinstance(bad_parsed, dict) or isinstance(good_parsed, dict),
            msg=f"Both bad/good parsing failed. bad={bad_raw[:400]} good={good_raw[:400]}",
        )

        self.records.append(
            {
                "comparison": {
                    "bad_schema_error_count": len(bad_errors),
                    "good_schema_error_count": len(good_errors),
                    "bad_final_score": bad_norm.get("final_score"),
                    "good_final_score": good_norm.get("final_score"),
                }
            }
        )


if __name__ == "__main__":
    unittest.main()
