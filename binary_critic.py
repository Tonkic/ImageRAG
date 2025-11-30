import base64
from PIL import Image
import io
import json
import os

# 复用 utils.py 中的基础辅助函数
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
        response_format={"type": "text"}, # Binary Critic 不需要复杂的 JSON mode
        temperature=temperature
    )

    res_text = res.choices[0].message.content
    return res_text

# -----------------------------------------------------------------------------
# Binary Critic (Baseline)
# 功能：仅判断生成结果是“成功”还是“失败”，不提供具体的错误类型或修正建议。
# -----------------------------------------------------------------------------

def retrieval_caption_generation(prompt, image_paths, gpt_client, model, decision=True, **kwargs):
    """
    Binary Critic 入口函数。
    为了兼容接口，保留了函数签名，但忽略了高级参数 (k_captions, k_concepts 等)。
    """

    # 1. 构造简单的二元判定 Prompt
    # 只要求回答 YES 或 NO
    msg = f"""
    Evaluate the attached image against the prompt: "{prompt}".
    Does the image accurately represent the prompt?

    You must respond with a SINGLE word: either "YES" or "NO".
    Do not provide any explanation.
    """

    # 2. 调用 VLM
    print(f"--- Binary Critic Running for: '{prompt}' ---")
    ans_text = message_gpt(msg, gpt_client, image_paths, model=model, images_idx=0)
    ans_clean = ans_text.strip().upper()

    print(f"--- Binary Critic Verdict: {ans_clean} ---")

    # 3. 构造标准返回格式 (兼容 SmartRAG 接口)
    if "YES" in ans_clean:
        return {
            "status": "success",
            "error_type": "none",
            "critique": "Image accepted by Binary Critic.",
            "features": [],
            "captions": []
        }
    else:
        # 失败情况
        # Binary Critic 不知道错在哪，所以 error_type 只能是 generic 的 "other"
        # 它也无法提供具体的 features，所以 captions 列表为空 (或者只是原始 prompt)
        return {
            "status": "error",
            "error_type": "other",  # 无法区分是 semantic 还是 attribute
            "critique": "Image rejected by Binary Critic (NO).",
            "features": [],
            "captions": [prompt] # 只能退回到使用原始 prompt 进行检索
        }