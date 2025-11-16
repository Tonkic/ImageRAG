import base64
from PIL import Image
import io
import json
import os
import torch


def convert_res_to_captions(res):
    """
    清理 VLM 返回的 caption 列表。
    """
    captions = [c.strip() for c in res.split("\n") if c != ""]
    for i in range(len(captions)):
        if captions[i][0].isnumeric() and captions[i][1] == ".":
            captions[i] = captions[i][2:]
        elif captions[i][0] == "-":
            captions[i] = captions[i][1:]
        elif f"{i+1}." in captions[i]:
            captions[i] = captions[i][captions[i].find(f"{i+1}."):]

        captions[i] = captions[i].strip().replace("'", "").replace('"', '')
    return captions

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
    向 VLM 发送消息的核心函数。
    """
    messages = [{"role": "user",
                 "content": [{"type": "text", "text": msg}]
                 }]
    if context_msgs:
        messages = context_msgs + messages

    if image_paths:
        base_64_images = [encode_image(image_path) for image_path in image_paths]
        for i, img in enumerate(base_64_images):
            if img: # 确保图像被成功编码
                messages[images_idx]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}})

    # (我们强制使用 text 模式，因为 API 不支持 json_mode)
    res = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "text"},
        temperature=temperature
    )

    res_text = res.choices[0].message.content
    return res_text


def message_gpt_w_error_handle(msg, client, image_paths, context_msgs, model, max_tries=3):
    """
    带错误处理的 message_gpt 包装器。
    """
    unable = True
    temp = 0
    concepts = "" # 初始化 concepts
    while unable and max_tries > 0:
        concepts = message_gpt(msg, client, image_paths, model=model, context_msgs=context_msgs, images_idx=0, temperature=temp)
        print("concepts from images", concepts)

        if "unable" not in concepts.lower() and "can't" not in concepts.lower():
            unable = False

        temp += 1.0 / max_tries # 确保浮点数除法
        max_tries -= 1

    if unable:
        print("was unable to generate concepts, using prompt as caption")
        return ""

    return concepts

# --------------------------------------------------
# --- 关键函数：VLM 诊断与 RAG 标题生成 ---
# --------------------------------------------------

def retrieval_caption_generation(prompt, image_paths, gpt_client, model, k_captions_per_concept=1, k_concepts=-1, decision=True, only_rephrase=False):
    """
    VLM 智能体函数：
    1. 诊断 V1 图像，返回 JSON 错误报告 (包含 "features" 列表)。
    2. 如果有错误，直接使用 "features" 列表作为 RAG 检索的 captions。
    """

    if decision:
        # --- 1. VLM 诊断 ---
        if len(image_paths) > 1:
            # (为多图编辑保留的备用分支)
            msg1 = f"""
            Analyze the attached images against the instruction: "{prompt}".
            The first image is the input, the second is the output.
            You MUST respond with a single JSON object.
            1. If the output image perfectly matches the instruction, respond:
            {{"status": "success", "error_type": "none", "critique": "Image is perfect.", "features": []}}
            2. If the output image fails, select an error_type and provide a critique and actionable features.
            {{"status": "error", "error_type": "wrong_concept", "critique": "The subject was not replaced correctly.", "features": ["subject from second image"]}}
            """
        else:
            # ！！！关键修改 1：新的 VLM 诊断提示！！！
            msg1 = f"""
            Analyze the attached image against the prompt: "{prompt}".
            You MUST respond with a single JSON object.

            1. If the image perfectly matches the prompt, respond:
            {{"status": "success", "error_type": "none", "critique": "Image is perfect.", "features": []}}

            2. If the image fails, you MUST choose one of the following error_types:
               - "missing_object": An object is missing.
               - "spatial_error": Objects are in the wrong place.
               - "count_error": The number of objects is wrong.
               - "text_error": Text in the image is incorrect or garbled.
               - "style_error": The artistic style is wrong.
               - "color_error": The color of an object is wrong.
               - "wrong_concept": A main subject is completely wrong.
               - "other": Any other general quality issue (blurry, deformed, etc.).

            3. **CRITICAL:** If you choose an error type, you MUST also provide a "features" array.
               - "features": A list of short, concrete, physical visual features *missing* or *wrong* in the image. (e.g., "three engines on the tail", "four engines under the wings", "a hat on the man's head", "five dogs", "the color red").

            Your JSON response MUST include "status", "error_type", "critique", and "features".

            Example 1 (Wrong Concept):
            {{"status": "error", "error_type": "wrong_concept", "critique": "The image shows an Airbus A320, not a Boeing 727-200.", "features": ["three engines on the tail", "T-shaped tail"]}}

            Example 2 (Missing Object):
            {{"status": "error", "error_type": "missing_object", "critique": "The man is not wearing a hat.", "features": ["a red baseball cap on the man's head"]}}

            Example 3 (Count Error):
            {{"status": "error", "error_type": "count_error", "critique": "The prompt asked for 5 dogs, but the image only shows 3.", "features": ["five dogs", "5 dogs"]}}
            """

        ans_text = message_gpt(msg1, gpt_client, image_paths, model=model, images_idx=0)
        print(f"--- VLM Decision Check --- The model's raw answer was: '{ans_text}'")

        try:
            # 清理 VLM 可能返回的 markdown 标记
            if "```json" in ans_text:
                ans_text = ans_text.split("```json")[1].split("```")[0].strip()
            # 如果没有 markdown，尝试从文本中提取第一个 JSON 对象
            elif "{" in ans_text and "}" in ans_text:
                start_index = ans_text.find("{")
                end_index = ans_text.rfind("}") + 1
                ans_text = ans_text[start_index:end_index]

            response_json = json.loads(ans_text)
        except Exception as e:
            print(f"[Error] VLM did not return valid JSON: {e}. Defaulting to RAG.")
            response_json = {"status": "error", "error_type": "other", "critique": "VLM response was not valid JSON.", "features": []}

        if response_json.get("status") == "success":
            return response_json # 返回 {"status": "success", ...}

        # --- 2. 状态为 "error"，处理 RAG 标题 ---

        # ！！！关键修改 2：使用 "features" 列表！！！
        features = response_json.get("features", [])

        if features:
            # 如果 VLM 提供了具体的 "features"，直接使用它们作为 RAG 标题
            print(f"VLM Features identified: {features}")
            response_json["captions"] = features # 直接使用特征列表
            return response_json

        # (回退逻辑：如果 VLM 懒惰了，没有提供 features)
        print("Warning: VLM did not provide 'features'. Falling back to critique-based caption generation.")
        critique = response_json.get("critique", "Image does not match prompt.")
        context_msgs = [{"role": "user", "content": [{"type": "text", "text": msg1}]},
                        {"role": "assistant", "content": [{"type": "text", "text": ans_text}]}]

        msg3 = (f'Based on the critique "{critique}", please suggest {k_captions_per_concept} image captions '
                'for images I can retrieve to fix this error. '
                'Only provide the captions, each on a new line. Do not add numbering.')

        captions_text = message_gpt(msg3, gpt_client, image_paths, model=model, context_msgs=context_msgs, images_idx=0)

        response_json["captions"] = convert_res_to_captions(captions_text)
        return response_json # 返回 {"status": "error", ..., "captions": [...]}

    else:
        # (旧的 'generation' 模式 - 您的新脚本不使用此分支)
        context_msgs = []
        msg2 = (f'What visual concepts does a generative model need to know to generate an image described by the prompt "{prompt}"? ...')
        concepts = message_gpt_w_error_handle(msg2, gpt_client, image_paths, context_msgs, model=model, max_tries=3)
        if concepts == "":
            return {"status": "error", "error_type": "other", "critique": "Failed to generate concepts in generation mode.", "features": [], "captions": []}

        print(f'retrieved concepts: {concepts}')

        msg3 = (f'For each concept you suggested above, please suggest {k_captions_per_concept} image captions...')
        context_msgs += [{"role": "user", "content": [{"type": "text", "text": msg2}]},
                         {"role": "assistant", "content": [{"type": "text", "text": concepts}]}]

        captions = message_gpt(msg3, gpt_client, image_paths, model=model, context_msgs=context_msgs, images_idx=0)

        return {
            "status": "error",
            "error_type": "other",
            "critique": "Generation mode initiated.",
            "features": [],
            "captions": convert_res_to_captions(captions)
        }