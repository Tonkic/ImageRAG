import json
from utils import message_gpt

def taxonomy_aware_diagnosis(prompt, image_paths, gpt_client, model):
    """
    SmartRAG V2.5: 增加了 Score 评分机制
    """
    msg = f"""
    You are an expert diagnostic agent for Text-to-Image generation. Analyze the attached image against the prompt: "{prompt}".

    You MUST respond with a single JSON object containing "status", "score", "error_type", "critique", and "features".

    ### 1. Scoring Criteria (0-10):
    - **10**: Perfect. Flawless alignment, high quality.
    - **8-9**: Excellent. Minor negligible flaws (e.g., tiny background artifact).
    - **6-7**: Acceptable. Concept correct, but has minor attribute/style errors.
    - **4-5**: Borderline. Major attribute errors (color/texture) or slight spatial issues.
    - **2-3**: Poor. Role swap, missing major objects, or significant spatial errors.
    - **0-1**: Failure. Wrong concept, hallucination, or garbage image.

    ### 2. Diagnosis Logic (Chain of Thought):
    1. Check Existence (Missing object?)
    2. Check Count (Wrong number?)
    3. Check Attributes (Wrong color/texture?)
    4. Check Role/Action (Subject/Object swapped?)
    5. Check Spatial (Wrong position?)
    6. Check Concept (Is it the right object?)

    ### 3. Select Error Type:
    Choose ONE: ["role_binding_error", "attribute_binding_error", "spatial_relation_error", "missing_object", "count_error", "wrong_concept", "text_error", "style_error", "other", "none"]
    (If score >= 9, error_type should be "none" and status "success").

    ### 4. Features Extraction:
    - "features": A list of specific visual elements that are MISSING or WRONG.

    Example JSON:
    {{
        "status": "error",
        "score": 4,
        "error_type": "role_binding_error",
        "critique": "The astronaut is riding the horse, but the prompt asked for the horse to ride the astronaut.",
        "features": ["astronaut riding horse"]
    }}
    """

    ans_text = message_gpt(msg, gpt_client, image_paths, model=model, images_idx=0)

    try:
        if "```json" in ans_text:
            ans_text = ans_text.split("```json")[1].split("```")[0].strip()
        elif "{" in ans_text:
            start = ans_text.find("{")
            end = ans_text.rfind("}") + 1
            ans_text = ans_text[start:end]

        response = json.loads(ans_text)

        # 容错处理：确保 score 存在且为整数
        if "score" not in response:
            response["score"] = 0
        else:
            response["score"] = int(response["score"])

        # 兼容旧逻辑：如果分数很高，强制设为 success
        if response["score"] >= 9:
            response["status"] = "success"
            response["error_type"] = "none"

        if "features" not in response:
            response["features"] = []
        response["captions"] = response["features"]

        return response

    except Exception as e:
        print(f"[TAC Error] JSON Parsing failed: {e}")
        return {
            "status": "error", "score": 0,
            "error_type": "other", "critique": "Parsing failed.",
            "features": [prompt], "captions": [prompt]
        }