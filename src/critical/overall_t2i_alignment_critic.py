from __future__ import annotations

from typing import Any, Dict, Optional

from src.critical.critic_common import call_vlm_json


def evaluate_overall_t2i_alignment(
    *,
    prompt: str,
    image_path: str,
    client: Any,
    model: str,
    domain: str = "aircraft",
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prompt_text = f"""
You are a text-image alignment evaluator for {domain} generation.

Prompt: "{prompt}".
First internally understand the image, then compare it against the prompt.

Return EXACTLY one JSON object with keys:
- image_description: string
- alignment_score: number in [0, 10]
- covered_elements: array of strings
- missing_elements: array of strings
- mismatched_elements: array of strings
- summary: string

No markdown. No prose outside JSON.
""".strip()

    return call_vlm_json(
        prompt=prompt_text,
        image_paths=[image_path],
        client=client,
        model=model,
        generation_kwargs=generation_kwargs,
        default={
            "image_description": "",
            "alignment_score": 0.0,
            "covered_elements": [],
            "missing_elements": [prompt],
            "mismatched_elements": [],
            "summary": "parse_failed",
        },
    )


__all__ = ["evaluate_overall_t2i_alignment"]
