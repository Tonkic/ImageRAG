from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.critical.critic_common import DOMAIN_ATTRIBUTE_AXES, call_vlm_json


def evaluate_fine_grained_alignment(
    *,
    prompt: str,
    image_path: str,
    client: Any,
    model: str,
    domain: str = "aircraft",
    attributes: Optional[List[str]] = None,
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    domain = (domain or "aircraft").lower()
    attributes = attributes or DOMAIN_ATTRIBUTE_AXES.get(domain, DOMAIN_ATTRIBUTE_AXES["aircraft"])
    attr_text = "\n".join(f"- {x}" for x in attributes)

    prompt_text = f"""
You are a domain expert for {domain} fine-grained recognition.

Evaluate whether the image matches the prompt: "{prompt}".
Check these fine-grained attributes one by one:
{attr_text}

Return EXACTLY one JSON object with keys:
- target_entity: string
- detected_entity: string
- attribute_checks: array of objects {{attribute: string, match: boolean, evidence: string}}
- fine_grained_score: number in [0, 10]
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
            "target_entity": prompt,
            "detected_entity": "",
            "attribute_checks": [],
            "fine_grained_score": 0.0,
            "summary": "parse_failed",
        },
    )


__all__ = ["evaluate_fine_grained_alignment"]
