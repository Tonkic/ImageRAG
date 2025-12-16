import json
import base64
import io
import os
import re  # Added for robust regex extraction
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
    # [Change] Initialize with empty content list to allow flexible ordering if needed
    # But for now, keep text first.
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

        # [Debug] Check if images are actually added
        if not base_64_images:
            print("[TAC Warning] No images successfully encoded!")

        # [Fix] Insert images BEFORE text. Many VLMs (including GPT-4o and Qwen-VL) handle [Image, Text] better.
        # Current structure: messages[images_idx]["content"] is [{"type": "text", ...}]

        for i, img in enumerate(base_64_images):
            # Insert at the beginning (index 0)
            messages[images_idx]["content"].insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"}
            })

    # [Debug] Print message structure (truncated)
    # print(f"[TAC Debug] Sending {len(messages)} messages. Content types: {[c['type'] for c in messages[images_idx]['content']]}")

    # [Debug] Print message structure (truncated)
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
                # response_format={"type": "text"}, # [Fix] Remove response_format for Qwen-VL compatibility
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

def retrieve_knowledge(prompt, client, model="gpt-4o"):
    """
    Knowledge Retrieval Agent ($A_{know}$)
    Retrieves key visual identification features for the subject in the prompt.
    Acts as a 'Sanity Check' or 'Hard Constraint' generator.
    """
    msg = f"""
    You are an expert Aircraft Encyclopedia.

    **Task:** Retrieve the key visual identification features for the aircraft mentioned in the prompt: "{prompt}".

    **Output Format:** A concise list of "Hard Constraints" that MUST be present.
    Focus on:
    1. Engine count and placement (e.g., 4 engines under wings).
    2. Wing configuration (e.g., Low-wing, swept-back).
    3. Tail configuration (e.g., T-tail, Conventional).
    4. Distinctive features (e.g., Hump, Winglets).

    Output as a plain text list. Do not include conversational filler.
    """

    return message_gpt(msg, client, [], model=model)

def taxonomy_aware_diagnosis(prompt, image_paths, gpt_client, model, reference_specs=None):
    """
    - 身份（Taxonomy）决定基准分 (0-6)
    - 细节（Details）决定上限分 (6-10)
    """

    # Inject Reference Specs if available
    specs_context = ""
    if reference_specs:
        specs_context = f"""
    ### Reference Specifications (Hard Constraints):
    The image MUST match these specs to be considered correct:
    {reference_specs}

    *Instruction:* Use these specs as the Ground Truth. If the image violates these (e.g., wrong engine count), it is a Taxonomy Error (Tier B or C).
    """

    msg = f"""
    You are an expert Aircraft Identification and Image Quality Assessor.

    **Task:** Analyze the generated image against the prompt: "{prompt}".
    {specs_context}

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
    - **error_analysis:**
        - "type": "Global" (Structural/Taxonomy error) OR "Local" (Detail/Text/Attribute error).

    ### Output JSON Format:
    {{
        "status": "success" | "error",
        "final_score": float,
        "taxonomy_check": "correct" | "wrong_subtype" | "wrong_object",
        "critique": "Brief explanation.",
        "refined_prompt": "...",
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

        # 1. Try Regex for ```json ... ```
        match = re.search(r"```json\s*(\{.*?\})\s*```", ans_text, re.DOTALL)
        if match:
            clean_text = match.group(1)
        else:
            # 2. Try Regex for first outer {...}
            # This regex finds the first { and the last }
            match = re.search(r"\{.*\}", ans_text, re.DOTALL)
            if match:
                clean_text = match.group(0)

        response = json.loads(clean_text)

        # --- Backward Compatibility Layer ---
        # Map new fields to old fields so existing scripts don't break
        response['score'] = response.get('final_score', 0)

        # Map retrieval_queries to needed_modifications for logging/legacy use
        response['needed_modifications'] = response.get('retrieval_queries', [])
        response['features'] = response.get('retrieval_queries', []) # Some scripts might use this

        # Merge needed_modifications into refined_prompt if not already present
        if response['needed_modifications']:
            modifications_str = ", ".join(response['needed_modifications'])
            # Check if modifications are already in refined_prompt to avoid duplication
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

def input_interpreter(prompt, client, model="gpt-4o"):
    """
    Stage 1: Input Interpreter Agent (A_in)
    Analyzes the user prompt to identify main subject, attributes, and fills in missing creative details.
    """
    msg = f"""
    You are an expert Input Interpreter Agent ($A_{{in}}$).
    Your goal is to transform a potentially vague user prompt into a structured analysis report.

    **User Prompt:** "{prompt}"

    **Tasks:**
    1. **Identify Elements:** Extract the 'main subject' and any explicit 'attributes' (livery, color, etc.).
    2. **Ambiguity Check:** Identify what is missing (background, lighting, view, specific livery if not stated).
    3. **Creativity Fills:** Autocomplete the missing details with high-quality, photorealistic, and context-appropriate descriptions.
       - If the subject is an aircraft, assume a standard or famous livery if not specified (e.g., Lufthansa for German planes, or House colors).
       - Set a realistic background (sky, runway).
       - Define lighting (natural, sunset, etc.) and composition (side view, dynamic angle).

    **Output JSON Format:**
    {{
      "Identified elements": {{
        "main subject": "...",
        "attributes": "..."
      }},
      "Creativity fills": {{
        "background": "...",
        "composition": "...",
        "lighting": "...",
        "visual style": "..."
      }},
      "Ambiguous elements": ["..."],
      "detailed_prompt": "Construct a full, highly detailed prompt combining all above elements for a text-to-image model."
    }}
    """

    ans_text = message_gpt(msg, client, [], model=model)

    try:
        clean_text = ans_text
        # Robust JSON extraction
        if "```json" in ans_text:
            parts = ans_text.split("```json")
            if len(parts) > 1:
                content = parts[1]
                if "```" in content:
                    clean_text = content.split("```")[0].strip()
                else:
                    clean_text = content.strip()

        if not clean_text or clean_text.strip() == "":
             if "{" in ans_text:
                start = ans_text.find("{")
                end = ans_text.rfind("}") + 1
                clean_text = ans_text[start:end]

        # [Fix] Handle "Extra data" error by attempting to parse only the valid JSON prefix
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                # Try to find the end of the first valid JSON object
                # Simple heuristic: count braces
                balance = 0
                end_idx = 0
                found_start = False
                for i, char in enumerate(clean_text):
                    if char == '{':
                        balance += 1
                        found_start = True
                    elif char == '}':
                        balance -= 1

                    if found_start and balance == 0:
                        end_idx = i + 1
                        break

                if end_idx > 0:
                    return json.loads(clean_text[:end_idx])
            raise e

    except Exception as e:
        print(f"[Input Interpreter Error] {e}")
        print(f"[Input Interpreter Debug] Raw response: {ans_text}")
        # Fallback
        return {
            "detailed_prompt": prompt,
            "Identified elements": {"main subject": prompt},
            "Creativity fills": {}
        }
