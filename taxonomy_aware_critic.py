import json
import base64
import io
import os
from PIL import Image

import time
import random

# --- Integrated Helper Functions ---
def encode_image(image_path, max_size=1024):
    """
    将图像文件编码为 base64 字符串，并限制最大尺寸以避免 API 错误。
    """
    try:
        image = Image.open(image_path)
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')

        # Resize if too large
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85) # Compress slightly
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def message_gpt(msg, client, image_paths=[], model="gpt-4o", context_msgs=[], images_idx=-1, temperature=0, max_retries=5):
    """
    向 VLM 发送消息的核心函数 (带重试机制)。
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

    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "text"},
                temperature=temperature
            )
            res_text = res.choices[0].message.content
            return res_text
        except Exception as e:
            print(f"[API Error] Attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Returning empty string.")
                return ""

def taxonomy_aware_diagnosis(prompt, image_paths, gpt_client, model):
    """
    - 身份（Taxonomy）决定基准分 (0-6)
    - 细节（Details）决定上限分 (6-10)
    """
    msg = f"""
    You are an expert Aircraft Identification and Image Quality Assessor.

    **Task:** Analyze the generated image against the prompt: "{prompt}".

    You MUST follow this strict **Tiered Scoring Protocol** to assign the `final_score`:

    ### 1. Taxonomy Check (The Gatekeeper) - Max 6.0 Points
    First, identify the object in the image. Does it match the specific aircraft model in the prompt?
    - **Tier C (Score 0-3):** Wrong Concept. (e.g., It's a bird, a truck, or a blurry mess).
    - **Tier B (Score 4-5):** Wrong Subtype/Variant. (e.g., Prompt asks for "707-320" [4 engines], but image shows a 2-engine plane like "737").
    - **Tier A (Score 6.0):** Correct Taxonomy. The image clearly shows the correct aircraft type (e.g., correct engine count, fuselage shape, wing configuration).
    *CRITICAL: If the taxonomy is wrong, the score CANNOT exceed 5.9, no matter how beautiful the image is.*

    ### 2. Detail & Aesthetic Bonus - Max +4.0 Points
    ONLY if the image reached Tier A (Score >= 6.0), add bonus points for details:
    - **+1.0 Structure:** Accurate landing gear, tail fin shape, winglets.
    - **+1.0 Attributes:** Correct livery/paint scheme, metallic texture, reflections.
    - **+1.0 Environment:** Realistic background (runway, sky) that matches the lighting.
    - **+1.0 Quality:** Sharp focus, 8k resolution, cinematic composition.

    ### 3. Remediation Strategy
    - **refined_prompt:** A full, descriptive prompt to generate a better version (Subject + Details + Aesthetics).
    - **retrieval_queries:** Short keywords to find reference images for the MISSING or WRONG parts.

    ### Output JSON Format:
    {{
        "status": "success" | "error",
        "final_score": float,  # Based on the logic above (e.g., 6.5, 8.2)
        "taxonomy_check": "correct" | "wrong_subtype" | "wrong_object",
        "critique": "Brief explanation of the score.",
        "refined_prompt": "...",
        "retrieval_queries": ["query1", "query2"]
    }}
    """

    ans_text = message_gpt(msg, gpt_client, image_paths, model=model, images_idx=0)

    try:
        clean_text = ans_text
        if "```json" in ans_text:
            clean_text = ans_text.split("```json")[1].split("```")[0].strip()
        elif "{" in ans_text:
            start = ans_text.find("{")
            end = ans_text.rfind("}") + 1
            clean_text = ans_text[start:end]

        response = json.loads(clean_text)

        # --- Backward Compatibility Layer ---
        # Map new fields to old fields so existing scripts don't break
        response['score'] = response.get('final_score', 0)

        tax_status = response.get('taxonomy_check', 'unknown')
        if tax_status == 'wrong_object':
            response['error_type'] = 'wrong_concept'
        elif tax_status == 'wrong_subtype':
            response['error_type'] = 'attribute_binding_error'
        elif tax_status == 'correct':
            response['error_type'] = 'none'
        else:
            response['error_type'] = 'other'

        # Map retrieval_queries to needed_modifications for logging/legacy use
        response['needed_modifications'] = response.get('retrieval_queries', [])
        response['features'] = response.get('retrieval_queries', []) # Some scripts might use this

        return response

    except Exception as e:
        print(f"[TAC Error] Parsing failed: {e}")
        return {
            "status": "error",
            "final_score": 0,
            "score": 0,
            "taxonomy_check": "error",
            "error_type": "other",
            "critique": f"Parsing error: {e}",
            "refined_prompt": prompt,
            "retrieval_queries": [prompt],
            "needed_modifications": [prompt]
        }
