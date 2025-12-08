import json
import base64
import io
import os
from PIL import Image

# --- Integrated Helper Functions ---
def encode_image(image_path):
    """
    将图像文件编码为 base64 字符串。
    """
    try:
        image = Image.open(image_path)
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def message_gpt(msg, client, image_paths=[], model="gpt-4o", context_msgs=[], images_idx=-1, temperature=0):
    """
    向 VLM 发送消息的核心函数 (标准版)。
    """
    messages = [{"role": "user",
                 "content": [{"type": "text", "text": msg}]
                 }]
    if context_msgs:
        messages = context_msgs + messages

    if image_paths:
        base_64_images = [encode_image(image_path) for image_path in image_paths]
        for i, img in enumerate(base_64_images):
            if img:
                messages[images_idx]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}})

    res = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "text"},
        temperature=temperature
    )

    res_text = res.choices[0].message.content
    return res_text

def taxonomy_aware_diagnosis(prompt, image_paths, gpt_client, model):
    msg = f"""
    You are an expert diagnostic agent for Text-to-Image generation. Analyze the attached image against the prompt: "{prompt}".

    You MUST respond with a single JSON object containing "status", "score", "error_type", "critique", "correct_features", "needed_modifications", and "actual_label".

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
    7. Identify Actual Object (If concept is wrong, what is it actually?)

    ### 3. Select Error Type:
    Choose ONE: ["role_binding_error", "attribute_binding_error", "spatial_relation_error", "missing_object", "count_error", "wrong_concept", "text_error", "style_error", "other", "none"]
    (If score >= 9, error_type should be "none" and status "success").

    ### 4. Features Analysis (Proactive):
    - "correct_features": List of visual elements from the prompt that are ALREADY CORRECT in the image.
    - "needed_modifications": List of concise, positive instructions to fix the errors. Describe what the image SHOULD look like.
      - Example: "Change the bird color to red" (instead of "The bird is blue").
      - Example: "Add wings" (instead of "Missing wings").
      - CRITICAL CONSTRAINT: Do NOT use specific model names, academic terms, or jargon (e.g., "DC-10", "Boeing 747") in this list. The T2I model likely does not know what a "DC-10" looks like by name.
      - REQUIRED: You MUST describe the VISUAL FEATURES that define the object using common concepts.
        - Bad: "Change the aircraft to a DC-10."
        - Good: "Add a third engine mounted at the base of the tail", "Ensure the fuselage is wide-bodied".
    - "actual_label": If the error is "wrong_concept" or "attribute_binding_error", provide the specific name/label of the object currently in the image (e.g., "Boeing 747", "Blue Jay"). If correct, return null.

    Example JSON:
    {{
        "status": "error",
        "score": 4,
        "error_type": "attribute_binding_error",
        "critique": "The bird is blue, but the prompt asked for a red bird.",
        "correct_features": ["bird shape", "tree branch"],
        "needed_modifications": ["make the bird red"],
        "actual_label": "Blue Bird"
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

        # [New] Ensure correct/modifications exist
        if "correct_features" not in response: response["correct_features"] = []
        if "needed_modifications" not in response: response["needed_modifications"] = []

        # [Compatibility]
        if "wrong_features" in response and not response["needed_modifications"]:
             response["needed_modifications"] = response["wrong_features"]

        response["captions"] = response["needed_modifications"]

        return response

    except Exception as e:
        print(f"[TAC Error] JSON Parsing failed: {e}")
        return {
            "status": "error", "score": 0,
            "error_type": "other", "critique": "Parsing failed.",
            "features": [prompt], "captions": [prompt],
            "correct_features": [], "needed_modifications": [prompt]
        }