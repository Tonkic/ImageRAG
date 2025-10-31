import base64
from PIL import Image
import io

def convert_res_to_captions(res):
    captions = [c.strip() for c in res.split("\n") if c != ""]
    for i in range(len(captions)):
        if captions[i][0].isnumeric() and captions[i][1] == ".":
            captions[i] = captions[i][2:]
        elif captions[i][0] == "-":
            captions[i] = captions[i][1:]
        elif f"{i+1}." in captions[i]:
            captions[i] = captions[i][captions[i].find(f"{i+1}.")+len(f"{i+1}."):]

        captions[i] = captions[i].strip().replace("'", "").replace('"', '')
    return captions

def encode_image(image_path):
    """
    Opens an image, cleans it by removing metadata, converts to RGB,
    and returns a base64 encoded string of the clean JPEG data.
    """
    try:
        # 1. 使用 Pillow 打开图像
        image = Image.open(image_path)

        # 2. 转换为 RGB 格式，确保兼容性并移除透明度通道
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')

        # 3. 将图像保存到内存中的 BytesIO 流。
        #    这个重新保存为 JPEG 的过程会自动剥离所有 EXIF 元数据。
        #    这是解决问题的关键步骤！
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        # 4. 对干净的图像字节流进行 Base64 编码
        return base64.b64encode(image_bytes).decode('utf-8')

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def message_gpt(msg, client, image_paths=[], model="gpt-4o", context_msgs=[], images_idx=-1, temperature=0):
    messages = [{"role": "user",
                 "content": [{"type": "text", "text": msg}]
                 }]
    if context_msgs:
        messages = context_msgs + messages

    if image_paths:
        base_64_images = [encode_image(image_path) for image_path in image_paths]
        for i, img in enumerate(base_64_images):
            messages[images_idx]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"}})

    res = client.chat.completions.create(
        # 关键修改：使用传入的 model 参数
        model=model,
        messages=messages,
        response_format={"type": "text"},
        temperature=temperature # for less randomness
    )

    res_text = res.choices[0].message.content
    return res_text

# 修改后的 message_gpt_w_error_handle
def message_gpt_w_error_handle(msg, client, image_paths, context_msgs, model, max_tries=3):
    unable = True
    temp = 0
    while unable and max_tries > 0:
        # 关键修改：将 model 参数传递给 message_gpt
        concepts = message_gpt(msg, client, image_paths, model=model, context_msgs=context_msgs, images_idx=0)
        print("concepts from images", concepts)

        if "unable" not in concepts and "can't" not in concepts:  # TODO make more generic
            unable = False

        temp += 1 / max_tries
        max_tries -= 1

    if unable:
        print("was unable to generate concepts, using prompt as caption")
        return ""

    return concepts

# 最终修改版的 retrieval_caption_generation
def retrieval_caption_generation(prompt, image_paths, gpt_client, model, k_captions_per_concept=1, k_concepts=-1, decision=True, only_rephrase=False):
    if decision:
        if len(image_paths) > 1:
            msg1 = f'Does the second image match the instruction "{prompt}" applied over the first one? consider both content and style aspects. only answer yes or no.'
        else:
            msg1 = f'Does this image match the prompt "{prompt}"? consider both content and style aspects. only answer yes or no.'

        # 修改点 1
        ans = message_gpt(msg1, gpt_client, image_paths, model=model)
        print(f"--- VLM Decision Check --- The model's raw answer was: '{ans}'")
        if 'yes' in ans.lower():
            return True

        context_msgs = [{"role": "user", "content": [{"type": "text", "text": msg1}]},
                        {"role": "assistant", "content": [{"type": "text", "text": ans}]}]

        print(f"Answer was {ans}. Running imageRAG")
        if only_rephrase:
            # 修改点 2
            rephrased_prompt = get_rephrased_prompt(prompt, gpt_client, image_paths, model=model, context_msgs=context_msgs, images_idx=0)
            print("rephrased_prompt:", rephrased_prompt)
            return rephrased_prompt

        msg2 = 'What are the differences between this image and the required prompt? ...' # 内容不变，省略
        if k_concepts > 0:
            msg2 += f'Return up to {k_concepts} concepts.'

        # 修改点 3
        concepts = message_gpt_w_error_handle(msg2, gpt_client, image_paths, context_msgs, model=model, max_tries=3)
        if concepts == "":
            return prompt
    else:  # generation mode
        context_msgs = []
        msg2 = (f'What visual concepts does a generative model need to know to generate an image described by the prompt "{prompt}"? ...') # 内容不变，省略

        # 修改点 4
        concepts = message_gpt_w_error_handle(msg2, gpt_client, image_paths, context_msgs, model=model, max_tries=3)
        if concepts == "":
            return prompt

    print(f'retrieved concepts: {concepts}')

    msg3 = (f'For each concept you suggested above, please suggest {k_captions_per_concept} image captions...') # 内容不变，省略
    context_msgs += [{"role": "user", "content": [{"type": "text", "text": msg2}]},
                     {"role": "assistant", "content": [{"type": "text", "text": concepts}]}]

    # 修改点 5
    captions = message_gpt(msg3, gpt_client, image_paths, model=model, context_msgs=context_msgs, images_idx=0)
    return captions


# 修改后的 get_rephrased_prompt
def get_rephrased_prompt(prompt, gpt_client, image_paths=[], model="gpt-4o", context_msgs=[], images_idx=-1):
    if not context_msgs:
        msg = f'Please rephrase the following prompt to make it clearer for a text-to-image generation model. If it\'s already clear, return it as it is. In your answer only provide the prompt and nothing else, and don\'t change the original meaning of the prompt. If it contains rare words, change the words to a description of their meaning. The prompt to be rephrased: "{prompt}"'
    else:
        msg = f'Please rephrase the following prompt to make it easier and clearer for the text-to-image generation model that generated the above image for this prompt. The goal is to generate an image that matches the given text prompt. If the prompt is already clear, return it as it is. Simplify and shorten long descriptions of known objects/entities but DO NOT change the original meaning of the text prompt. If the prompt contains rare words, change those words to a description of their meaning. In your answer only provide the prompt and nothing else. The prompt to be rephrased: "{prompt}"'

    # 关键修改：将 model 参数传递给 message_gpt
    ans = message_gpt(msg, gpt_client, image_paths, model=model, context_msgs=context_msgs, images_idx=images_idx)
    return ans.strip().replace('"', '').replace("'", '')