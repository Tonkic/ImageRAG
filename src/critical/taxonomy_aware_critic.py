import json
import base64
import io
import os
import ast
import re  # Added for robust regex extraction
from PIL import Image

import time
import random


def _extract_outer_json_block(text):
    if not text:
        return text
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return text[start:]


def _sanitize_json_like(text):
    fixed = (text or "").strip()
    fixed = fixed.replace("\u201c", '"').replace("\u201d", '"')
    fixed = fixed.replace("\u2018", "'").replace("\u2019", "'")
    fixed = fixed.replace("：", ":").replace("，", ",")
    fixed = fixed.replace("\\n", " ")
    fixed = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", fixed)
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    return fixed


def _extract_json_candidate(text):
    clean_text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", clean_text, re.DOTALL | re.IGNORECASE)
    if fenced:
        clean_text = fenced.group(1).strip()
    clean_text = _extract_outer_json_block(clean_text)
    return clean_text


def _parse_json_loose(text):
    candidates = []
    if text:
        candidates.append(text)
        candidates.append(_sanitize_json_like(text))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    for candidate in candidates:
        try:
            py_candidate = re.sub(r"\btrue\b", "True", candidate, flags=re.IGNORECASE)
            py_candidate = re.sub(r"\bfalse\b", "False", py_candidate, flags=re.IGNORECASE)
            py_candidate = re.sub(r"\bnull\b", "None", py_candidate, flags=re.IGNORECASE)
            parsed = ast.literal_eval(py_candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


def _extract_str(field_name, text):
    pattern = rf'["\']{field_name}["\']\s*[:=]\s*["\']([^"\']+)["\']'
    match = re.search(pattern, text or "", re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_float(field_name, text):
    pattern = rf'["\']{field_name}["\']\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)'
    match = re.search(pattern, text or "", re.IGNORECASE)
    return float(match.group(1)) if match else None


def _extract_list(field_name, text):
    pattern = rf'["\']{field_name}["\']\s*[:=]\s*\[(.*?)\]'
    match = re.search(pattern, text or "", re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    raw = match.group(1)
    parts = [p.strip().strip('"\' ') for p in raw.split(",") if p.strip()]
    return [p for p in parts if p]


def _get_alias_value(data, aliases):
    if not isinstance(data, dict):
        return None

    for alias in aliases:
        if alias in data:
            return data[alias]

    lower_map = {
        str(k).lower(): k for k in data.keys() if isinstance(k, str)
    }
    for alias in aliases:
        found = lower_map.get(alias.lower())
        if found is not None:
            return data[found]

    return None


def _to_float_safe(value, default=0.0):
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?[0-9]+(?:\.[0-9]+)?", value)
        if match:
            try:
                return float(match.group(0))
            except Exception:
                return default
    return default


def _canon_key(key):
    if not isinstance(key, str):
        return key
    key = key.strip().lower().replace("-", "_").replace(" ", "_")
    key = re.sub(r"_+", "_", key)
    while key.startswith("_"):
        key = key[1:]
    return key


def _normalize_keys_deep(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[_canon_key(k)] = _normalize_keys_deep(v)
        return out
    if isinstance(obj, list):
        return [_normalize_keys_deep(x) for x in obj]
    return obj


def _normalize_tac_contract_keys(response):
    if not isinstance(response, dict):
        return response

    data = _normalize_keys_deep(response)
    alias = {
        "finalscore": "final_score",
        "score": "final_score",
        "taxonomycheck": "taxonomy_check",
        "taxonomy": "taxonomy_check",
        "retrievalquery": "retrieval_queries",
        "retrievalqueries": "retrieval_queries",
        "queries": "retrieval_queries",
        "retrieved_negative_prompt": "refined_negative_prompt",
        "refined_negative_prompts": "refined_negative_prompt",
        "refine_negative_prompt": "refined_negative_prompt",
        "erroranalysis": "error_analysis",
        "analysis": "critique",
        "reason": "critique",
    }
    for src, dst in alias.items():
        if src in data and dst not in data:
            data[dst] = data[src]

    return data


def _validate_tac_schema(response, compact_output=False):
    errors = []
    if not isinstance(response, dict):
        return ["not_dict"]

    if compact_output:
        required = {
            "final_score": (int, float),
            "retrieval_queries": list,
            "refined_prompt": str,
        }
    else:
        required = {
            "status": str,
            "final_score": (int, float),
            "taxonomy_check": str,
            "critique": str,
            "refined_prompt": str,
            "retrieval_queries": list,
            "error_analysis": dict,
            "refined_negative_prompt": str,
        }
    for key, typ in required.items():
        if key not in response:
            errors.append(f"missing_key:{key}")
            continue
        if not isinstance(response[key], typ):
            errors.append(f"wrong_type:{key}:{type(response[key]).__name__}")

    if not compact_output and isinstance(response.get("taxonomy_check"), str):
        if response["taxonomy_check"] not in {"correct", "wrong_subtype", "wrong_object", "error"}:
            errors.append(f"invalid_taxonomy:{response['taxonomy_check']}")
    if isinstance(response.get("retrieval_queries"), list):
        if not all(isinstance(x, str) for x in response["retrieval_queries"]):
            errors.append("retrieval_queries_not_all_str")
    if not compact_output and isinstance(response.get("error_analysis"), dict):
        et = response["error_analysis"].get("type")
        if et is not None and et not in {"Global", "Local"}:
            errors.append(f"invalid_error_analysis_type:{et}")

    return errors


def _build_tac_repair_prompt(schema_errors, previous_output, original_prompt):
    error_text = ", ".join(schema_errors[:12]) if schema_errors else "unknown_schema_error"
    prev = (previous_output or "")[:1400]
    return (
        "Your previous TAC JSON output is invalid.\n"
        f"Schema errors: {error_text}\n"
        f"Previous output: {prev}\n\n"
        "Now output ONLY valid JSON and follow this instruction exactly:\n"
        f"{original_prompt}"
    )


def _normalize_taxonomy_check(raw_value):
    allowed = {"correct", "wrong_subtype", "wrong_object", "error"}

    if isinstance(raw_value, bool):
        return "correct" if raw_value else "error"

    if isinstance(raw_value, list):
        if not raw_value:
            return "error"
        return _normalize_taxonomy_check(raw_value[0])

    if raw_value is None:
        return "error"

    value = str(raw_value).strip().lower()
    if value in allowed:
        return value

    if value in {"true", "yes", "right", "pass", "passed", "valid", "ok"}:
        return "correct"

    if "subtype" in value or "variant" in value or "model" in value:
        return "wrong_subtype"

    if "wrong" in value or "incorrect" in value or "object" in value:
        return "wrong_object"

    return "error"


def _infer_taxonomy_from_score(final_score):
    if final_score >= 6.0:
        return "correct"
    if final_score >= 4.0:
        return "wrong_subtype"
    if final_score > 0.0:
        return "wrong_object"
    return "error"


def _normalize_error_analysis(raw_value, taxonomy_check):
    if isinstance(raw_value, dict):
        if isinstance(raw_value.get("type"), str) and raw_value["type"] in {"Global", "Local"}:
            return raw_value
        normalized_type = "Global" if taxonomy_check != "correct" else "Local"
        fixed = dict(raw_value)
        fixed["type"] = normalized_type
        return fixed

    if isinstance(raw_value, str):
        raw = raw_value.strip().lower()
        if raw == "global":
            return {"type": "Global"}
        if raw == "local":
            return {"type": "Local"}

    normalized_type = "Global" if taxonomy_check != "correct" else "Local"
    return {"type": normalized_type}


def _normalize_tac_response(response, prompt):
    if not isinstance(response, dict):
        response = {}

    schema_warnings = []

    raw_status = _get_alias_value(response, ["status"])
    status_unrecognized = False
    if raw_status is None:
        status = "success"
        schema_warnings.append("status_missing_defaulted")
    else:
        status_l = str(raw_status).strip().lower()
        if status_l in {"success", "ok", "valid"}:
            status = "success"
        elif status_l in {"partial_success", "partial"}:
            status = "partial_success"
        elif status_l in {"error", "failed", "fail"}:
            status = "error"
        else:
            status = "success"
            status_unrecognized = True
            schema_warnings.append(f"status_unrecognized:{raw_status}")

    raw_score = _get_alias_value(response, ["final_score", "score"])
    final_score = _to_float_safe(raw_score, default=0.0)
    if 0.0 < final_score <= 1.0:
        final_score = final_score * 10.0
        schema_warnings.append("score_rescaled_0_1_to_0_10")
    final_score = max(0.0, min(10.0, float(final_score)))

    raw_taxonomy = _get_alias_value(response, ["taxonomy_check", "taxonomy", "taxonomyCheck"])
    if raw_taxonomy in (None, ""):
        taxonomy_check = _infer_taxonomy_from_score(final_score)
        schema_warnings.append("taxonomy_inferred_from_score")
    else:
        taxonomy_check = _normalize_taxonomy_check(raw_taxonomy)
    if taxonomy_check == "error" and raw_taxonomy not in (None, "", "error"):
        schema_warnings.append(f"taxonomy_normalized_from:{raw_taxonomy}")

    if (raw_status is None or status_unrecognized) and taxonomy_check == "error":
        status = "error"

    raw_critique = _get_alias_value(response, ["critique", "analysis", "reason"])
    critique = str(raw_critique).strip() if raw_critique is not None else ""

    raw_refined_prompt = _get_alias_value(response, ["refined_prompt", "refinedPrompt", "prompt"])
    refined_prompt = str(raw_refined_prompt).strip() if raw_refined_prompt is not None else ""
    if not refined_prompt:
        refined_prompt = prompt
        schema_warnings.append("refined_prompt_missing_defaulted")

    raw_queries = _get_alias_value(
        response,
        [
            "retrieval_queries",
            "retrievalQueries",
            "queries",
            "needed_modifications",
            "features",
        ],
    )
    if isinstance(raw_queries, str):
        queries = [raw_queries]
    elif isinstance(raw_queries, list):
        queries = raw_queries
    else:
        queries = []
    queries = [str(q).strip() for q in queries if str(q).strip()]

    raw_negative_prompt = _get_alias_value(
        response,
        [
            "refined_negative_prompt",
            "refine_negative_prompt",
            "refinedNegativePrompt",
            "negative_prompt",
        ],
    )
    refined_negative_prompt = "" if raw_negative_prompt is None else str(raw_negative_prompt).strip()

    raw_error_analysis = _get_alias_value(response, ["error_analysis", "errorAnalysis", "error"])
    error_analysis = _normalize_error_analysis(raw_error_analysis, taxonomy_check)

    response["status"] = status
    response["final_score"] = final_score
    response["score"] = final_score
    response["taxonomy_check"] = taxonomy_check
    response["critique"] = critique
    response["refined_prompt"] = refined_prompt
    response["retrieval_queries"] = queries
    response["needed_modifications"] = queries
    response["features"] = queries
    response["refined_negative_prompt"] = refined_negative_prompt
    response["error_analysis"] = error_analysis
    response["schema_warnings"] = schema_warnings

    if response.get("needed_modifications") and response.get("refined_prompt"):
        modifications_str = ", ".join(response["needed_modifications"])
        if modifications_str and modifications_str.lower() not in response["refined_prompt"].lower():
            response["refined_prompt"] = f"{response['refined_prompt']}. Ensure the following details are correct: {modifications_str}."

    return response

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

def message_gpt(msg, client, image_paths=[], model="gpt-4o", context_msgs=[], images_idx=-1, temperature=0, max_retries=5, **generation_kwargs):
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
                temperature=temperature,
                **generation_kwargs,
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

def taxonomy_aware_diagnosis(prompt, image_paths, gpt_client, model, reference_specs=None, domain="aircraft", vlm_generation_kwargs=None, repair_retries=1, compact_output=False):

    domain = domain.lower()

    # Inject Reference Specs if available
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

    base_msg_full = f"""
    You are {persona}.

    **Task:** Analyze the generated image against the prompt: "{prompt}".
    {specs_context}

    You MUST follow this strict **Tiered Scoring Protocol** to assign the `final_score`:

    ### 1. Taxonomy Check (The Gatekeeper) - Max 6.0 Points
    First, identify the object in the image. Does it match the specific subject in the prompt?
    - **Tier C (Score 1.0-3.9):** {tier_c_desc} (Give 1.0-2.0 for total failures, 3.0-3.9 for somewhat related objects).
    - **Tier B (Score 4.0-5.9):** {tier_b_desc} (Give 4.0-5.0 if loosely matching but wrong subvariant, 5.1-5.9 if very close but missing 1-2 key specs).
    - **Tier A (Score 6.0):** {tier_a_desc}
    *CRITICAL: If the taxonomy is wrong, the score CANNOT exceed 5.9, no matter how beautiful the image is. DO NOT just give 0.0 unless the image is completely blank or corrupt.*

    ### 2. Detail & Aesthetic Bonus - Max +4.0 Points
    ONLY if the image reached Tier A (Score >= 6.0), add bonus points for details:
    - **+1.0 Structure:** {bonus_structure}
    - **+1.0 Attributes:** {bonus_attributes}
    - **+1.0 Environment:** {bonus_env}
    - **+1.0 Quality:** Sharp focus, high resolution, cinematic composition.

    ### 3. Remediation Strategy
    - **refined_prompt:** A full, descriptive prompt to generate a better version (Subject + Details + Aesthetics).
    - **retrieval_queries:** Short keywords to find reference images for the MISSING or WRONG parts.
    - **error_analysis:**
        - "type": "Global" (Structural/Taxonomy error) OR "Local" (Detail/Text/Attribute error).

    ### STRICT OUTPUT CONTRACT (MUST FOLLOW EXACTLY)
    Return EXACTLY one JSON object. No prose. No markdown. No code fences.

    Allowed keys ONLY (exact spelling, lower snake_case, no leading/trailing spaces):
    - status
    - final_score
    - taxonomy_check
    - critique
    - refined_prompt
    - retrieval_queries
    - error_analysis
    - refined_negative_prompt

    Type constraints:
    - status: string, one of ["success", "error"]
    - final_score: number in [0, 10]
    - taxonomy_check: string, one of ["correct", "wrong_subtype", "wrong_object", "error"]
    - critique: string
    - refined_prompt: string
    - retrieval_queries: array of strings
    - error_analysis: object with key "type" in ["Global", "Local"]
    - refined_negative_prompt: string, <= 15 words

    Forbidden:
    - Any alias keys (e.g., retrievalQueries, refine_negative_prompt, taxonomy, score)
    - Any keys with spaces (e.g., " status ")
    - Any non-string taxonomy_check (no bool/list)

    If uncertain, STILL output valid defaults:
    {{
      "status": "error",
      "final_score": 0,
      "taxonomy_check": "error",
      "critique": "",
      "refined_prompt": "{prompt}",
      "retrieval_queries": ["{prompt}"],
      "error_analysis": {{"type": "Global"}},
      "refined_negative_prompt": ""
    }}
    """

    base_msg_compact = f"""
    You are {persona}.

    Task: Analyze the generated image against prompt "{prompt}".
    {specs_context}

    Scoring rule:
    - 0-3.9: wrong concept/object
    - 4.0-5.9: wrong subtype/variant
    - 6.0-10.0: correct taxonomy (+detail bonuses)

    Return EXACTLY one JSON object with ONLY these 3 keys:
    - final_score (number in [0, 10])
    - retrieval_queries (array of strings)
    - refined_prompt (string)

    No prose. No markdown. No code fences. No extra keys.

    If uncertain, output defaults:
    {{
        "final_score": 0,
        "retrieval_queries": ["{prompt}"],
        "refined_prompt": "{prompt}"
    }}
    """

    def _fallback_from_text(ans_text_local):
        clean_text = _extract_json_candidate(ans_text_local)
        final_score = _extract_float("final_score", clean_text) or 0.0
        refined_prompt = _extract_str("refined_prompt", clean_text) or prompt
        queries = _extract_list("retrieval_queries", clean_text) or _extract_list("retrievalQueries", clean_text) or []
        fallback = {
            "final_score": final_score,
            "refined_prompt": refined_prompt,
            "retrieval_queries": queries,
        }
        if not compact_output:
            fallback.update({
                "status": "partial_success",
                "taxonomy_check": "error",
                "critique": "JSON malformed, used field-level fallback.",
                "error_analysis": {"type": "Global"},
                "refined_negative_prompt": "",
            })
        return fallback

    def _error_result(reason):
        return {
            "status": "error",
            "final_score": 0,
            "score": 0,
            "taxonomy_check": "error",
            "critique": reason,
            "refined_prompt": prompt,
            "retrieval_queries": [prompt],
            "needed_modifications": [prompt],
            "features": [prompt],
            "refined_negative_prompt": "",
            "error_analysis": {"type": "Global"},
            "schema_warnings": ["tac_fallback_error_result"],
        }

    gen_kwargs = dict(vlm_generation_kwargs or {})
    temperature = gen_kwargs.pop("temperature", 0)
    current_msg = base_msg_compact if compact_output else base_msg_full
    last_normalized = _error_result("TAC did not produce valid schema output.")

    for attempt in range(max(0, int(repair_retries)) + 1):
        ans_text = message_gpt(
            current_msg,
            gpt_client,
            image_paths,
            model=model,
            images_idx=0,
            temperature=temperature,
            **gen_kwargs,
        )

        if not ans_text:
            if attempt < int(repair_retries):
                current_msg = _build_tac_repair_prompt(["empty_response"], "", base_msg_compact if compact_output else base_msg_full)
                continue
            return _error_result("Empty response from VLM.")

        try:
            clean_text = _extract_json_candidate(ans_text)
            response = _parse_json_loose(clean_text)

            if response is None:
                print("[TAC Info] Attempting aggressive field-level recovery...")
                response = _fallback_from_text(ans_text)

            if not isinstance(response, dict):
                response = _fallback_from_text(ans_text)

            normalized_keys_response = _normalize_tac_contract_keys(response)
            schema_errors = _validate_tac_schema(normalized_keys_response, compact_output=compact_output)

            normalized = _normalize_tac_response(normalized_keys_response, prompt)
            last_normalized = normalized

            if not schema_errors:
                return normalized

            if attempt < int(repair_retries):
                current_msg = _build_tac_repair_prompt(schema_errors, ans_text, base_msg_compact if compact_output else base_msg_full)
                continue

            normalized.setdefault("schema_warnings", [])
            normalized["schema_warnings"].append(
                f"schema_retry_exhausted:{'|'.join(schema_errors[:6])}"
            )
            return normalized

        except Exception as e:
            preview = (ans_text or "")[:400].replace("\n", "\\n")
            print(f"[TAC Error] Unexpected parse error: {e}")
            print(f"[TAC Debug] Raw response preview: {preview}")
            if attempt < int(repair_retries):
                current_msg = _build_tac_repair_prompt([f"parse_exception:{e}"], ans_text, base_msg_compact if compact_output else base_msg_full)
                continue
            return _error_result(f"Parsing error: {e}")

    return last_normalized

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
        clean_text = _extract_json_candidate(ans_text)
        parsed = _parse_json_loose(clean_text)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Unable to parse Input Interpreter response into dict")

    except Exception as e:
        print(f"[Input Interpreter Error] {e}")
        preview = (ans_text or "")[:400].replace("\n", "\\n")
        print(f"[Input Interpreter Debug] Raw response preview: {preview}")
        # Fallback
        return {
            "detailed_prompt": prompt,
            "Identified elements": {"main subject": prompt},
            "Creativity fills": {}
        }
