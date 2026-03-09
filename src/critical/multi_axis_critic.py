from __future__ import annotations

from typing import Any, Dict, Optional

from src.critical.critic_common import (
    DOMAIN_ATTRIBUTE_AXES,
    DOMAIN_DIMENSIONS,
    get_domain_eval_dimensions,
)
from src.critical.fine_grained_alignment_critic import evaluate_fine_grained_alignment
from src.critical.identity_preservation_critic import evaluate_identity_preservation
from src.critical.overall_t2i_alignment_critic import evaluate_overall_t2i_alignment
from src.critical.visual_realism_critic import evaluate_visual_realism


def run_multi_axis_critic(
    *,
    prompt: str,
    image_path: str,
    client: Any,
    model: str,
    domain: str = "aircraft",
    reference_image_path: Optional[str] = None,
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = {
        "domain": domain,
        "dimensions": get_domain_eval_dimensions(domain),
        "fine_grained_alignment": evaluate_fine_grained_alignment(
            prompt=prompt,
            image_path=image_path,
            client=client,
            model=model,
            domain=domain,
            generation_kwargs=generation_kwargs,
        ),
        "visual_realism": evaluate_visual_realism(
            prompt=prompt,
            image_path=image_path,
            client=client,
            model=model,
            domain=domain,
            generation_kwargs=generation_kwargs,
        ),
        "overall_t2i_alignment": evaluate_overall_t2i_alignment(
            prompt=prompt,
            image_path=image_path,
            client=client,
            model=model,
            domain=domain,
            generation_kwargs=generation_kwargs,
        ),
    }

    if reference_image_path:
        result["identity_preservation"] = evaluate_identity_preservation(
            prompt=prompt,
            generated_image_path=image_path,
            reference_image_path=reference_image_path,
            client=client,
            model=model,
            domain=domain,
            generation_kwargs=generation_kwargs,
        )
    else:
        result["identity_preservation"] = None

    return result


__all__ = [
    "DOMAIN_DIMENSIONS",
    "DOMAIN_ATTRIBUTE_AXES",
    "get_domain_eval_dimensions",
    "evaluate_fine_grained_alignment",
    "evaluate_identity_preservation",
    "evaluate_visual_realism",
    "evaluate_overall_t2i_alignment",
    "run_multi_axis_critic",
]
