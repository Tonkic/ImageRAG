from __future__ import annotations

from typing import Any, Dict, Optional

from src.critical.critic_common import call_vlm_json


def evaluate_identity_preservation(
    *,
    prompt: str,
    generated_image_path: str,
    reference_image_path: str,
    client: Any,
    model: str,
    domain: str = "aircraft",
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prompt_text = f"""
You are a {domain} identity matching judge.

Image-1 is the retrieved real reference image.
Image-2 is the generated image.
The text target is: "{prompt}".

Decide whether Image-2 preserves the same fine-grained identity as Image-1 and still matches the text target.

Return EXACTLY one JSON object with keys:
- target_entity: string
- reference_entity: string
- generated_entity: string
- same_identity: boolean
- identity_score: number in [0, 10]
- mismatch_cues: array of strings
- summary: string

No markdown. No prose outside JSON.
""".strip()

    return call_vlm_json(
        prompt=prompt_text,
        image_paths=[reference_image_path, generated_image_path],
        client=client,
        model=model,
        generation_kwargs=generation_kwargs,
        default={
            "target_entity": prompt,
            "reference_entity": "",
            "generated_entity": "",
            "same_identity": False,
            "identity_score": 0.0,
            "mismatch_cues": [],
            "summary": "parse_failed",
        },
    )


__all__ = ["evaluate_identity_preservation"]
