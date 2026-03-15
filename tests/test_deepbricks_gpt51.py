import json
import os
import unittest
from pathlib import Path

import httpx
import openai


DEEPBRICKS_BASE_URL = os.environ.get("DEEPBRICKS_BASE_URL", "https://api.deepbricks.ai/v1/")
DEEPBRICKS_MODEL = os.environ.get("DEEPBRICKS_MODEL", "gpt-5.1")
DEEPBRICKS_TIMEOUT = float(os.environ.get("DEEPBRICKS_TIMEOUT", "120"))
DEFAULT_PROXY = os.environ.get("DEEPBRICKS_DEFAULT_PROXY", "http://127.0.0.1:10000")
KEY_FILE = Path(__file__).with_name(".gpt_api_key")
PROXY_ENV_KEYS = [
    "DEEPBRICKS_PROXY",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
]


def _read_api_key() -> str | None:
    if not KEY_FILE.is_file():
        return None
    value = KEY_FILE.read_text(encoding="utf-8").strip()
    if not value or value.startswith("#"):
        return None
    return value


def _resolve_proxy() -> str | None:
    for key in PROXY_ENV_KEYS:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return DEFAULT_PROXY


def _extract_json_text(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
    return text


def _build_aircraft_prompt() -> str:
    prompt = "a photo of a Boeing 707-320 parked on the runway at sunset, side view, photorealistic"
    return f"""You are an expert Input Interpreter Agent specialized in aircraft identification and aviation photography.
Decompose the following prompt for a RAG-based text-to-image pipeline.

**User Prompt:** \"{prompt}\"

Output ONLY one valid JSON object (no prose, no markdown):
{{
  \"high_importance\": {{\"entity\": \"<exact subject name>\", \"key_features\": [\"<f1>\", \"<f2>\"]}},
  \"low_importance\":  {{\"background\": \"...\", \"composition\": \"...\", \"lighting\": \"...\", \"visual_style\": \"...\"}},
  \"importance_weights\": {{\"entity_identity\": 0.9, \"structural_detail\": 0.8, \"environment\": 0.3, \"artistic_style\": 0.2}},
  \"retrieval_query\": \"a photo of a <entity>\",
  \"generation_prompt\": \"<full detailed prompt for T2I>\",
  \"ambiguous_elements\": []
}}

STRICT:
- entity must be the exact subject name from prompt.
- retrieval_query must use the format: \"a photo of a <entity>\".
- generation_prompt should combine identity, structure, lighting, background, and style.
- Return EXACTLY one JSON object."""


def _validate_aircraft_pipeline_schema(data: dict) -> list[str]:
    errors = []
    required_top_keys = {
        "high_importance",
        "low_importance",
        "importance_weights",
        "retrieval_query",
        "generation_prompt",
        "ambiguous_elements",
    }
    missing = sorted(required_top_keys - set(data.keys()))
    if missing:
        errors.append(f"missing_top_keys:{missing}")

    high = data.get("high_importance")
    if not isinstance(high, dict):
        errors.append("high_importance_not_dict")
    else:
        if not isinstance(high.get("entity"), str) or not high.get("entity", "").strip():
            errors.append("high_importance.entity_invalid")
        feats = high.get("key_features")
        if not isinstance(feats, list) or not all(isinstance(x, str) for x in feats):
            errors.append("high_importance.key_features_invalid")

    low = data.get("low_importance")
    if not isinstance(low, dict):
        errors.append("low_importance_not_dict")
    else:
        for key in ["background", "composition", "lighting", "visual_style"]:
            if not isinstance(low.get(key), str):
                errors.append(f"low_importance.{key}_invalid")

    weights = data.get("importance_weights")
    if not isinstance(weights, dict):
        errors.append("importance_weights_not_dict")
    else:
        for key in ["entity_identity", "structural_detail", "environment", "artistic_style"]:
            value = weights.get(key)
            if not isinstance(value, (int, float)):
                errors.append(f"importance_weights.{key}_invalid")

    retrieval_query = data.get("retrieval_query")
    if not isinstance(retrieval_query, str) or not retrieval_query.startswith("a photo of a "):
        errors.append("retrieval_query_format_invalid")

    if not isinstance(data.get("generation_prompt"), str):
        errors.append("generation_prompt_invalid")
    if not isinstance(data.get("ambiguous_elements"), list):
        errors.append("ambiguous_elements_invalid")

    return errors


def _print_response(label: str, response) -> None:
    print(f"\n=== {label} ===")
    if hasattr(response, "model_dump_json"):
        print(response.model_dump_json(indent=2))
    else:
        print(response)


class TestDeepbricksGPT51(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("RUN_DEEPBRICKS_REAL_TEST") != "1":
            raise unittest.SkipTest("Set RUN_DEEPBRICKS_REAL_TEST=1 to run the real Deepbricks GPT-5.1 test.")

        api_key = _read_api_key()
        if not api_key:
            raise unittest.SkipTest(f"Deepbricks API key not found in {KEY_FILE}")

        proxy = _resolve_proxy()
        cls.api_key = api_key
        cls.proxy = proxy
        cls.http_client = httpx.Client(proxy=proxy, timeout=DEEPBRICKS_TIMEOUT, trust_env=True)
        cls.client = openai.OpenAI(
            api_key=api_key,
            base_url=DEEPBRICKS_BASE_URL,
            http_client=cls.http_client,
        )

    @classmethod
    def tearDownClass(cls):
        http_client = getattr(cls, "http_client", None)
        if http_client is not None:
            http_client.close()

    def test_chat_completion_basic(self):
        response = self.client.chat.completions.create(
            model=DEEPBRICKS_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Reply with exactly OK."},
            ],
            max_tokens=16,
        )

        _print_response("basic", response)

        self.assertTrue(response.choices, "Deepbricks returned no choices")
        message = response.choices[0].message
        text = ((getattr(message, "content", None) or "").strip() or "")

        self.assertTrue(text, "Deepbricks returned empty content")
        self.assertIn("OK", text.upper())

        usage = getattr(response, "usage", None)
        self.assertIsNotNone(usage, "Usage info should be returned for a real call")

    def test_chat_completion_json_mode(self):
        response = self.client.chat.completions.create(
            model=DEEPBRICKS_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": "Return JSON: {\"status\": \"ok\", \"provider\": \"deepbricks\"}."},
            ],
            response_format={"type": "json_object"},
            max_tokens=64,
        )

        _print_response("json_mode", response)

        self.assertTrue(response.choices, "Deepbricks returned no choices")
        message = response.choices[0].message
        text = _extract_json_text((getattr(message, "content", None) or "").strip() or "")

        self.assertTrue(text.startswith("{"), f"Expected JSON object, got: {text!r}")
        data = json.loads(text)
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("provider"), "deepbricks")

    def test_chat_completion_aircraft_example(self):
        response = self.client.chat.completions.create(
            model=DEEPBRICKS_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": _build_aircraft_prompt()},
            ],
            response_format={"type": "json_object"},
            max_tokens=256,
        )

        _print_response("aircraft_example", response)

        self.assertTrue(response.choices, "Deepbricks returned no choices")
        message = response.choices[0].message
        text = _extract_json_text((getattr(message, "content", None) or "").strip() or "")
        data = json.loads(text)
        errors = _validate_aircraft_pipeline_schema(data)

        self.assertFalse(errors, f"Aircraft pipeline schema errors: {errors}; response={data!r}")
        self.assertIn("707-320", data["high_importance"]["entity"])
        self.assertIn("707-320", data["retrieval_query"])


if __name__ == "__main__":
    unittest.main()
