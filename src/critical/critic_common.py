from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.critical.taxonomy_aware_critic import (
    _extract_json_candidate,
    _parse_json_loose,
    encode_image,
)


DOMAIN_DIMENSIONS: Dict[str, Dict[str, str]] = {
    "cub": {
        "fine_grained_alignment": "部位级属性是否与文本一致，例如头冠颜色、翅膀斑纹、尾羽形状、喙形状。",
        "identity_preservation": "生成图是否仍像检索参考图对应的那个具体鸟类亚种，而不是相近鸟类。",
        "visual_realism": "是否存在伪影，如多余翅膀、扭曲腿部、不合理羽毛边界。",
        "overall_t2i_alignment": "主体、姿态、背景环境是否覆盖原始 Prompt 的主要描述。",
    },
    "aircraft": {
        "fine_grained_alignment": "型号级属性是否一致，例如发动机数量/位置、尾翼构型、翼尖、机身轮廓、涂装关键特征。",
        "identity_preservation": "生成图是否仍像检索参考图对应的那个具体机型/子型号，而不是相邻型号。",
        "visual_realism": "是否存在伪影，如机翼/起落架结构扭曲、发动机数量不合理、透视不合常识。",
        "overall_t2i_alignment": "主体机型、姿态、场景、背景元素是否覆盖原始 Prompt 描述。",
    },
}


DOMAIN_ATTRIBUTE_AXES: Dict[str, List[str]] = {
    "cub": [
        "head color or crest",
        "beak shape and size",
        "wing pattern or wing bar",
        "breast or belly color",
        "tail shape or tail color",
        "leg color",
    ],
    "aircraft": [
        "aircraft subtype / variant identity",
        "engine count and placement",
        "wing configuration or wingtip shape",
        "tail configuration",
        "fuselage profile / nose shape",
        "livery or markings",
    ],
}


def get_domain_eval_dimensions(domain: str) -> Dict[str, str]:
    domain = (domain or "").lower()
    return DOMAIN_DIMENSIONS.get(domain, DOMAIN_DIMENSIONS["aircraft"])


def make_vlm_messages(prompt: str, image_paths: List[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for image_path in image_paths:
        img_base64 = encode_image(image_path)
        if img_base64 is None:
            raise ValueError(f"Failed to encode image: {image_path}")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
            }
        )
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def call_vlm_json(
    *,
    prompt: str,
    image_paths: List[str],
    client: Any,
    model: str,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    default: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    generation_kwargs = dict(generation_kwargs or {})
    temperature = generation_kwargs.pop("temperature", 0.35)

    messages = make_vlm_messages(prompt, image_paths)
    res = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **generation_kwargs,
    )
    raw = res.choices[0].message.content if res and res.choices else ""

    parsed = _parse_json_loose(_extract_json_candidate(raw))
    if isinstance(parsed, dict):
        parsed.setdefault("_raw_response", raw)
        return parsed

    fallback = dict(default or {})
    fallback["_raw_response"] = raw
    fallback["_parse_failed"] = True
    return fallback


__all__ = [
    "DOMAIN_DIMENSIONS",
    "DOMAIN_ATTRIBUTE_AXES",
    "get_domain_eval_dimensions",
    "make_vlm_messages",
    "call_vlm_json",
]
