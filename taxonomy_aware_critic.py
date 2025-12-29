import json
import base64
import io
import os
import re  # 用于更健壮的正则提取
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

        # 如果尺寸过大则调整大小
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85) # 轻微压缩
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def message_gpt(msg, client, image_paths=[], model="gpt-4o", context_msgs=[], images_idx=-1, temperature=0, max_retries=5):
    """
    向 VLM 发送消息的核心函数 (带重试机制)。
    """
    # 初始化为空内容列表，以便在需要时灵活排序
    # 但目前保持文本在前
    messages = [{"role": "user",
                 "content": [{"type": "text", "text": msg}]
                 }]
    if context_msgs:
        messages = context_msgs + messages

    if image_paths:
        base_64_images = []
        for image_path in image_paths:
            encoded = encode_image(image_path)
            if encoded:
                base_64_images.append(encoded)
            else:
                print(f"[TAC Warning] Failed to encode image: {image_path}")

        # [调试] 检查图像是否实际已添加
        if not base_64_images:
            print("[TAC Warning] No images successfully encoded!")

        # [修复] 在文本之前插入图像。许多 VLM（包括 GPT-4o 和 Qwen-VL）能更好地处理 [图像, 文本] 的顺序。
        # 当前结构：messages[images_idx]["content"] 是 [{"type": "text", ...}]

        for i, img in enumerate(base_64_images):
            # 插入到开头（索引 0）
            messages[images_idx]["content"].insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"}
            })

    # [调试] 打印消息结构（截断）
    try:
        content_types = [c.get('type', 'unknown') for c in messages[images_idx]['content']]
        print(f"[TAC Debug] Sending to {model}. Content types: {content_types}")
    except Exception as e:
        print(f"[TAC Debug] Error inspecting messages: {e}")

    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                # response_format={"type": "text"}, # [修复] 为了兼容 Qwen-VL 移除 response_format
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

def generate_knowledge_specs(prompt, client, model="gpt-4o", domain="aircraft"):
    """
    根据提示词和领域生成知识规格（Hard Constraints）。
    """
    domain = domain.lower()

    if domain == "aircraft":
        persona = "an expert Aircraft Encyclopedia"
        focus_instructions = """
    1. Engine count and placement (e.g., 4 engines under wings).
    2. Wing configuration (e.g., Low-wing, swept-back).
    3. Tail configuration (e.g., T-tail, Conventional).
    4. Distinctive features (e.g., Hump, Winglets)."""

    elif domain in ["birds", "cub", "cub_200_2011"]:
        persona = "an expert Ornithologist"
        focus_instructions = """
    1. Beak shape, length, and color.
    2. Plumage patterns (e.g., stripes, spots) and specific colors on head, wings, breast, and tail.
    3. Distinctive markings (e.g., eye rings, crests, wing bars).
    4. Leg color and tail shape."""

    else:
        persona = "an expert Visual Encyclopedia"
        focus_instructions = """
    1. Key structural components and overall shape.
    2. Distinctive colors, patterns, textures, or materials.
    3. Identifying markers, logos, or unique features.
    4. Relative size and proportions."""

    msg = f"""
    You are {persona}.

    **Task:** Retrieve the key visual identification features for the subject mentioned in the prompt: "{prompt}".

    **Output Format:** A concise list of "Hard Constraints" that MUST be present.
    Focus on:
    {focus_instructions}

    Output as a plain text list. Do not include conversational filler.
    """

    return message_gpt(msg, client, [], model=model)

def taxonomy_aware_diagnosis(prompt, image_paths, gpt_client, model, reference_specs=None, domain="aircraft"):
    """
    - 身份（Taxonomy）决定基准分 (0-6)
    - 细节（Details）决定上限分 (6-10)
    """

    domain = domain.lower()

    # 如果可用，注入参考规格
    specs_context = ""
    if reference_specs:
        specs_context = f"""
    ### Reference Specifications (Hard Constraints):
    The image MUST match these specs to be considered correct:
    {reference_specs}

    *Instruction:* Use these specs as the Ground Truth. If the image violates these (e.g., wrong engine count), it is a Taxonomy Error (Tier B or C).
    """

    if domain == "aircraft":
        persona = "an expert Aircraft Identification and Image Quality Assessor"
        tier_c_desc = "Wrong Concept. (e.g., It's a bird, a truck, or a blurry mess)."
        tier_b_desc = "Wrong Subtype/Variant. (e.g., Prompt asks for \"707-320\" [4 engines], but image shows a 2-engine plane like \"737\")."
        tier_a_desc = "Correct Taxonomy. The image clearly shows the correct aircraft type (e.g., correct engine count, fuselage shape, wing configuration)."
        bonus_structure = "Accurate landing gear, tail fin shape, winglets."
        bonus_attributes = "Correct livery/paint scheme, metallic texture, reflections."
        bonus_env = "Realistic background (runway, sky) that matches the lighting."
    elif domain in ["birds", "cub", "cub_200_2011"]:
        persona = "an expert Ornithologist and Bird Image Quality Assessor"
        tier_c_desc = "Wrong Concept. (e.g., It's an airplane, a cat, or a blurry mess)."
        tier_b_desc = "Wrong Species/Subspecies. (e.g., Prompt asks for \"Red-winged Blackbird\", but image shows a generic blackbird without red patches)."
        tier_a_desc = "Correct Taxonomy. The image clearly shows the correct bird species (e.g., correct beak shape, plumage patterns, distinctive markings)."
        bonus_structure = "Accurate beak shape, eye rings, leg color."
        bonus_attributes = "Correct feather texture, vibrant colors, realistic plumage."
        bonus_env = "Realistic natural habitat (tree branch, water, forest) that matches the lighting."
    else:
        persona = "an expert Image Quality Assessor"
        tier_c_desc = "Wrong Concept. (e.g., The image does not depict the subject in the prompt)."
        tier_b_desc = "Wrong Variant/Type. (e.g., The subject looks similar but misses key specific features requested)."
        tier_a_desc = "Correct Taxonomy. The image clearly shows the correct subject with all key identifying features."
        bonus_structure = "Accurate structural details and proportions."
        bonus_attributes = "Correct colors, textures, and materials."
        bonus_env = "Realistic and appropriate background/environment."

    msg = f"""
    You are {persona}.

    **Task:** Analyze the generated image against the prompt: "{prompt}".
    {specs_context}

    **CRITICAL INSTRUCTION: Chain-of-Thought Inspection**
    Do NOT jump to conclusions. You must inspect the image step-by-step BEFORE assigning a score.

    **Step 1: Visual Inspection (Mental Scratchpad)**
    - Inspect the key features visible in the image.
    - For Aircraft: Check Wing Tips (Straight vs Winglets?), Engines (Cigar vs Fat?), Fuselage (Hump vs Smooth?).
    - For Birds: Check Beak, Plumage, Markings.
    - Identify the subject shown in the image based *only* on visual evidence.

    **Step 2: Comparison**
    - Compare your findings from Step 1 with the Target Subject in the prompt (and Reference Specs).
    - Does the observed subject match the target?

    **Step 3: Scoring (Tiered Protocol)**
    ### 1. Taxonomy Check (The Gatekeeper) - Max 6.0 Points
    - **Tier C (Score 0-3):** {tier_c_desc}
    - **Tier B (Score 4-5):** {tier_b_desc}
    - **Tier A (Score 6.0):** {tier_a_desc}
    *If the taxonomy is wrong (Tier B/C), the score CANNOT exceed 5.9.*

    ### 2. Detail & Aesthetic Bonus - Max +4.0 Points
    (Only if Score >= 6.0)
    - **+1.0 Structure:** {bonus_structure}
    - **+1.0 Attributes:** {bonus_attributes}
    - **+1.0 Environment:** {bonus_env}
    - **+1.0 Quality:** Sharp focus, high resolution.

    **Step 4: Remediation (Positive Assertions)**
    - **refined_prompt:** A full, descriptive prompt.
    - **CRITICAL:** Avoid "Pink Elephant" negative constraints (e.g., "No winglets", "No hump").
    - **ACTION:** Convert negative constraints into POSITIVE visual descriptions.
        - Bad: "No winglets" -> Good: "Standard straight wing tips"
        - Bad: "No hump" -> Good: "Smooth, straight fuselage roof"
        - Bad: "No modern engines" -> Good: "Long, thin cigar-shaped engines"

    ### Output JSON Format:
    {{
        "status": "success" | "error",
        "final_score": float,
        "taxonomy_check": "correct" | "wrong_subtype" | "wrong_object",
        "critique": "Step-by-step analysis: 1. Observed [Feature X]... 2. This contradicts target [Feature Y]... 3. Conclusion...",
        "refined_prompt": "...",
        "concise_retrieval_prompt": "...",
        "retrieval_queries": ["..."],
        "error_analysis": {{
            "type": "Global" | "Local"
        }}
    }}
    """

    ans_text = message_gpt(msg, gpt_client, image_paths, model=model, images_idx=0)

    if not ans_text:
        print("[TAC Error] Empty response from VLM.")
        return {
            "status": "error",
            "final_score": 0,
            "score": 0,
            "taxonomy_check": "error",
            "critique": "Empty response from VLM.",
            "refined_prompt": prompt,
            "retrieval_queries": [prompt],
            "needed_modifications": [prompt]
        }

    try:
        clean_text = ans_text.strip()

        # 1. 尝试使用正则提取 ```json ... ```
        match = re.search(r"```json\s*(\{.*?\})\s*```", ans_text, re.DOTALL)
        if match:
            clean_text = match.group(1)
        else:
            # 2. 尝试正则提取最外层的 {...}
            # 此正则查找第一个 { 和最后一个 }
            match = re.search(r"\{.*\}", ans_text, re.DOTALL)
            if match:
                clean_text = match.group(0)

        response = json.loads(clean_text)

        # --- 向后兼容层 ---
        # 将新字段映射到旧字段，以免破坏现有脚本
        response['score'] = response.get('final_score', 0)

        # 将 retrieval_queries 映射到 needed_modifications 以供日志记录/旧版使用
        response['needed_modifications'] = response.get('retrieval_queries', [])
        response['features'] = response.get('retrieval_queries', []) # 某些脚本可能会使用此字段

        # 如果 refined_prompt 中尚未包含 needed_modifications，则将其合并进去
        if response['needed_modifications']:
            modifications_str = ", ".join(response['needed_modifications'])
            # 检查 modifications 是否已在 refined_prompt 中，以避免重复
            if modifications_str.lower() not in response['refined_prompt'].lower():
                 response['refined_prompt'] = f"{response['refined_prompt']}. Ensure the following details are correct: {modifications_str}."

        return response

    except Exception as e:
        print(f"[TAC Error] Parsing failed: {e}")
        print(f"[TAC Debug] Raw response: {ans_text}")
        return {
            "status": "error",
            "final_score": 0,
            "score": 0,
            "taxonomy_check": "error",
            "critique": f"Parsing error: {e}",
            "refined_prompt": prompt,
            "retrieval_queries": [prompt],
            "needed_modifications": [prompt]
        }

