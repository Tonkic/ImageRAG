import argparse
import json
import os
import sys
import textwrap
import time
import urllib.error
import urllib.request
from typing import Optional

import openai


DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1/"
DEFAULT_MODEL = "Qwen/Qwen3.5-397B-A17B"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_JSON_SYSTEM_PROMPT = "You are a helpful assistant designed to output JSON."
API_PRESETS = {
    "kimi_k25": {
        "base_url": DEFAULT_BASE_URL,
        "model": "Pro/moonshotai/Kimi-K2.5",
    },
}
DEFAULT_KEY_FILES = [
    os.path.join(os.path.dirname(__file__), ".text_api_key"),
    os.path.join(os.path.dirname(__file__), ".api_key"),
    os.path.expanduser("~/.imageRAG_text_api_key"),
]


def disable_proxy_env() -> None:
    for key in [
        "http_proxy", "https_proxy", "all_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    ]:
        os.environ.pop(key, None)


def read_api_key_from_local_file() -> Optional[str]:
    for path in DEFAULT_KEY_FILES:
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    value = f.read().strip()
                if value and not value.startswith("#"):
                    return value
        except Exception:
            continue
    return None


def build_aircraft_input_prompt(user_prompt: str) -> str:
    return textwrap.dedent(
        f"""
        You are an expert Input Interpreter Agent specialized in aircraft identification and aviation photography.

        **Task:** Decompose the following prompt into importance-weighted components for a
        text-to-image generation pipeline with Retrieval-Augmented Generation (RAG).

        **User Prompt:** "{user_prompt}"

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

        **STRICT OUTPUT CONTRACT (MUST FOLLOW):**
        - Return EXACTLY one JSON object. No prose, no markdown, no code fence.
        - Keys must be exact lower snake_case names above (no leading/trailing spaces).
        - If uncertain, still return valid defaults:
          - high_importance.entity: "{user_prompt}"
          - retrieval_query: "a photo of a [entity]"
          - generation_prompt: "{user_prompt}"
        """
    ).strip()


def build_prompt(scenario: str) -> str:
    prompts = {
        "hello": "Say hello in exactly one English sentence.",
        "aircraft_input": build_aircraft_input_prompt(
            "a photo of a Boeing 707-320 parked on the runway at sunset, side view, photorealistic"
        ),
        "aircraft_rewrite": (
            "Rewrite this aircraft generation prompt to strongly emphasize identity fidelity and visible structure: \n"
            "'a photo of a Boeing 707-320, photorealistic'.\n"
            "Output only the rewritten prompt in English."
        ),
    }
    return prompts[scenario]


def build_client(api_key: str, base_url: str) -> openai.OpenAI:
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def get_message_text(message) -> str:
    content = (getattr(message, "content", None) or "").strip()
    if content:
        return content
    reasoning = (getattr(message, "reasoning_content", None) or "").strip()
    return reasoning


def resolve_system_prompt(system_prompt: Optional[str], json_mode: bool) -> Optional[str]:
    if json_mode and (system_prompt is None or system_prompt == DEFAULT_SYSTEM_PROMPT):
        return DEFAULT_JSON_SYSTEM_PROMPT
    return system_prompt


def validate_aircraft_input_schema(result_obj) -> list[str]:
    errors = []
    if not isinstance(result_obj, dict):
        return ["not_dict"]

    required_top_keys = {
        "high_importance",
        "low_importance",
        "importance_weights",
        "retrieval_query",
        "generation_prompt",
        "ambiguous_elements",
    }
    actual_top_keys = set(result_obj.keys())
    missing = sorted(required_top_keys - actual_top_keys)
    extra = sorted(actual_top_keys - required_top_keys)
    if missing:
        errors.append(f"missing_top_keys:{missing}")
    if extra:
        errors.append(f"extra_top_keys:{extra}")

    high = result_obj.get("high_importance")
    if not isinstance(high, dict):
        errors.append("high_importance_not_dict")
    else:
        if not isinstance(high.get("entity"), str) or not high.get("entity", "").strip():
            errors.append("high_importance.entity_invalid")
        if not isinstance(high.get("key_features"), list):
            errors.append("high_importance.key_features_not_list")
        elif not all(isinstance(x, str) for x in high.get("key_features", [])):
            errors.append("high_importance.key_features_items_not_str")

    low = result_obj.get("low_importance")
    if not isinstance(low, dict):
        errors.append("low_importance_not_dict")
    else:
        for key in ["background", "composition", "lighting", "visual_style"]:
            if not isinstance(low.get(key), str):
                errors.append(f"low_importance.{key}_not_str")

    weights = result_obj.get("importance_weights")
    if not isinstance(weights, dict):
        errors.append("importance_weights_not_dict")
    else:
        for key in ["entity_identity", "structural_detail", "environment", "artistic_style"]:
            value = weights.get(key)
            if not isinstance(value, (int, float)):
                errors.append(f"importance_weights.{key}_not_number")
            elif not 0.0 <= float(value) <= 1.0:
                errors.append(f"importance_weights.{key}_out_of_range")

    if not isinstance(result_obj.get("retrieval_query"), str):
        errors.append("retrieval_query_not_str")
    if not isinstance(result_obj.get("generation_prompt"), str):
        errors.append("generation_prompt_not_str")
    ambiguous = result_obj.get("ambiguous_elements")
    if not isinstance(ambiguous, list):
        errors.append("ambiguous_elements_not_list")
    elif not all(isinstance(x, str) for x in ambiguous):
        errors.append("ambiguous_elements_items_not_str")

    return errors


def report_schema_check(scenario: str, text: str) -> None:
    if scenario != "aircraft_input":
        return
    try:
        result_obj = json.loads(text)
    except Exception as e:
        print("schema_valid:", False)
        print("schema_errors:", [f"invalid_json:{e}"])
        print()
        return

    errors = validate_aircraft_input_schema(result_obj)
    print("schema_valid:", not errors)
    print("schema_errors:", errors)
    print()


def run_chat_probe(
    client: openai.OpenAI,
    *,
    model: str,
    scenario: str,
    prompt: str,
    max_tokens: int,
    disable_thinking: bool,
    timeout: int,
    system_prompt: Optional[str],
    json_mode: bool,
) -> None:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    if disable_thinking:
        kwargs["extra_body"] = {"enable_thinking": False}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    text = get_message_text(msg)

    mode_name = "no-thinking" if disable_thinking else "default"
    print(f"=== chat probe: {mode_name} ===")
    print("model:", model)
    print("content:", repr(getattr(msg, "content", None)))
    print("reasoning_content:", repr(getattr(msg, "reasoning_content", None)))
    print("resolved_text:", repr(text))
    print("usage:", getattr(resp, "usage", None))
    print()
    report_schema_check(scenario, text)

    if not text:
        raise RuntimeError(f"chat probe returned empty text in {mode_name} mode")


def run_http_probe(
    *,
    api_key: str,
    base_url: str,
    model: str,
    scenario: str,
    prompt: str,
    max_tokens: int,
    disable_thinking: bool,
    timeout: int,
    system_prompt: Optional[str],
    json_mode: bool,
) -> None:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if disable_thinking:
        payload["enable_thinking"] = False
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    mode_name = "no-thinking" if disable_thinking else "default"
    print(f"=== http probe: {mode_name} ===")
    print("endpoint:", endpoint)
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e
    elapsed = time.time() - start

    data = json.loads(raw)
    msg = (((data.get("choices") or [{}])[0]).get("message") or {})
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    text = content or reasoning

    print(f"elapsed_sec: {elapsed:.2f}")
    print("content:", repr(content))
    print("reasoning_content:", repr(reasoning))
    print("resolved_text:", repr(text))
    print("usage:", data.get("usage"))
    print()
    report_schema_check(scenario, text)

    if not text:
        raise RuntimeError(f"http probe returned empty text in {mode_name} mode")


def run_models_probe(client: openai.OpenAI) -> None:
    print("=== models probe ===")
    try:
        models = client.models.list()
        data = list(getattr(models, "data", []) or [])
        print("models_count:", len(data))
        preview = [getattr(x, "id", None) for x in data[:10]]
        print("models_preview:", preview)
        print()
    except Exception as e:
        print(f"models probe skipped: {e}")
        print()


def resolve_api_key(cli_key: Optional[str]) -> str:
    api_key = (
        cli_key
        or os.environ.get("TEXT_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or read_api_key_from_local_file()
    )
    if not api_key:
        raise SystemExit(
            "Missing API key. Use --api-key, set TEXT_API_KEY / OPENAI_API_KEY, or place the key in tests/.text_api_key."
        )
    return api_key


def resolve_endpoint_config(args: argparse.Namespace) -> tuple[str, str]:
    preset = API_PRESETS.get(args.preset) if args.preset else None
    base_url = preset["base_url"] if preset else DEFAULT_BASE_URL
    model = preset["model"] if preset else DEFAULT_MODEL

    if args.base_url:
        base_url = args.base_url
    if args.model:
        model = args.model

    return base_url, model


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe SiliconFlow/OpenAI-compatible text API.")
    parser.add_argument("--api-key", type=str, default=None, help="API key. Prefer env var instead.")
    parser.add_argument("--preset", type=str, default=None, choices=sorted(API_PRESETS.keys()),
                        help="Named API preset for base_url/model. Explicit --base-url/--model override preset values.")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="hello",
                        choices=["hello", "aircraft_input", "aircraft_rewrite"])
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt. If omitted, uses the selected scenario template.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--skip-models", action="store_true", help="Skip models.list probe.")
    parser.add_argument("--only-no-thinking", action="store_true",
                        help="Only run enable_thinking=False mode to avoid long thinking latency.")
    parser.add_argument("--transport", type=str, default="sdk",
                        choices=["sdk", "http", "both"],
                        help="Use OpenAI SDK, raw HTTP, or both.")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--json-mode", action="store_true",
                        help="Enable response_format={\"type\": \"json_object\"}.")
    parser.add_argument("--no-json-mode", action="store_true",
                        help="Disable JSON mode even for structured scenarios.")
    parser.add_argument("--no-proxy", action="store_true",
                        help="Temporarily disable HTTP(S)_PROXY / ALL_PROXY for this probe.")
    args = parser.parse_args()

    if args.json_mode and args.no_json_mode:
        raise SystemExit("--json-mode and --no-json-mode cannot be used together.")

    if args.no_proxy:
        disable_proxy_env()

    api_key = resolve_api_key(args.api_key)
    base_url, model = resolve_endpoint_config(args)
    client = build_client(api_key, base_url)
    prompt = args.prompt or build_prompt(args.scenario)
    json_mode = args.json_mode or (args.scenario == "aircraft_input" and not args.no_json_mode)
    system_prompt = resolve_system_prompt(args.system_prompt, json_mode)

    print("preset:", args.preset)
    print("base_url:", base_url)
    print("model:", model)
    print("scenario:", args.scenario)
    print("json_mode:", json_mode)
    print("system_prompt:", repr(system_prompt))
    print("prompt:", repr(prompt))
    print()

    if not args.skip_models:
        run_models_probe(client)

    if args.transport in ["sdk", "both"]:
        if not args.only_no_thinking:
            run_chat_probe(
                client,
                model=model,
                scenario=args.scenario,
                prompt=prompt,
                max_tokens=args.max_tokens,
                disable_thinking=False,
                timeout=args.timeout,
                system_prompt=system_prompt,
                json_mode=json_mode,
            )
        run_chat_probe(
            client,
            model=model,
            scenario=args.scenario,
            prompt=prompt,
            max_tokens=args.max_tokens,
            disable_thinking=True,
            timeout=args.timeout,
            system_prompt=system_prompt,
            json_mode=json_mode,
        )

    if args.transport in ["http", "both"]:
        if not args.only_no_thinking:
            run_http_probe(
                api_key=api_key,
                base_url=base_url,
                model=model,
                scenario=args.scenario,
                prompt=prompt,
                max_tokens=args.max_tokens,
                disable_thinking=False,
                timeout=args.timeout,
                system_prompt=system_prompt,
                json_mode=json_mode,
            )
        run_http_probe(
            api_key=api_key,
            base_url=base_url,
            model=model,
            scenario=args.scenario,
            prompt=prompt,
            max_tokens=args.max_tokens,
            disable_thinking=True,
            timeout=args.timeout,
            system_prompt=system_prompt,
            json_mode=json_mode,
        )

    print("API probe passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"API probe failed: {e}", file=sys.stderr)
        raise
