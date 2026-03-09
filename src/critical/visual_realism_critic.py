from __future__ import annotations

from typing import Any, Dict, Optional

from src.critical.critic_common import call_vlm_json


def evaluate_visual_realism(
    *,
    prompt: str,
    image_path: str,
    client: Any,
    model: str,
    domain: str = "aircraft",
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prompt_text = f"""
You are a strict visual realism critic for {domain} images.

Inspect the image for prompt: "{prompt}".
Focus on physical plausibility and generation artifacts.

Return EXACTLY one JSON object with keys:
- realism_score: number in [0, 10]
- artifact_detected: boolean
- structural_issues: array of strings
- texture_issues: array of strings
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
            "realism_score": 0.0,
            "artifact_detected": True,
            "structural_issues": ["parse_failed"],
            "texture_issues": [],
            "summary": "parse_failed",
        },
    )


__all__ = ["evaluate_visual_realism"]
