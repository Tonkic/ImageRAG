import os
import torch
import clip
from open_clip import create_model_from_pretrained, get_tokenizer
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm

# [Optional] Long-CLIP Imports
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Long-CLIP'))
try:
    from model import longclip
    LONGCLIP_AVAILABLE = True
except ImportError:
    LONGCLIP_AVAILABLE = False
    print("Warning: Long-CLIP not found.")

# -----------------------------------------------------------------------------
# Static Retrieval (Baseline)
# 功能：仅使用预训练模型 (CLIP/SigLIP) 进行基于余弦相似度的 Top-K 检索。
# -----------------------------------------------------------------------------

def get_clip_similarities(prompts, image_paths, embeddings_path="", bs=1024, k=50, device='cuda:0'):
    """
    使用 CLIP (ViT-B/32) 计算文本与图像库的相似度，并返回 Top-K 结果。
    """
    # 加载模型
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 确保 prompt 是列表
    if isinstance(prompts, str):
        prompts = [prompts]

    # [Fix] Add truncate=True
    text = clip.tokenize(prompts, truncate=True).to(device)

    all_scores = []
    all_paths = []

    with torch.no_grad():
        # 1. 编码文本
        text_features = model.encode_text(text)
        normalized_text_vectors = F.normalize(text_features, p=2, dim=1)

        # 2. 批量处理图像
        # 使用切片确保处理所有数据，包括最后一个不足 bs 的 batch
        for bi in range(0, len(image_paths), bs):

            # 缓存文件命名
            cache_file = os.path.join(embeddings_path, f"clip_embeddings_b{bi}.pt")
            current_batch_paths = image_paths[bi : bi + bs]

            if os.path.exists(cache_file):
                # 加载缓存
                try:
                    data = torch.load(cache_file, map_location=device, weights_only=True)
                    cached_emb = data['normalized_clip_embeddings']
                    # Dimension Check
                    if cached_emb.shape[-1] == normalized_text_vectors.shape[-1]:
                        normalized_im_vectors = cached_emb
                        final_bi_paths = data.get('paths', current_batch_paths)
                    else:
                        print(f"Warning: Cached CLIP embeddings in {cache_file} have dimension {cached_emb.shape[-1]}, but model has {normalized_text_vectors.shape[-1]}. Ignoring cache.")
                        normalized_im_vectors = None
                except Exception as e:
                    print(f"Error loading cache {cache_file}: {e}")
                    normalized_im_vectors = None

            if not os.path.exists(cache_file) or normalized_im_vectors is None:
                # 计算 Embedding
                images = []
                valid_paths = []

                for path in current_batch_paths:
                    try:
                        # 强制转 RGB，防止灰度图报错
                        img_pil = Image.open(path).convert("RGB")
                        image = preprocess(img_pil).unsqueeze(0).to(device)
                        images.append(image)
                        valid_paths.append(path)
                    except Exception as e:
                        print(f"Warning: Could not read {path}: {e}")
                        continue

                if not images:
                    continue

                images = torch.stack(images).squeeze(1)
                image_features = model.encode_image(images)
                normalized_im_vectors = F.normalize(image_features, p=2, dim=1)
                final_bi_paths = valid_paths

                # 保存缓存
                if embeddings_path:
                    os.makedirs(embeddings_path, exist_ok=True)
                    torch.save({
                        "normalized_clip_embeddings": normalized_im_vectors,
                        "paths": final_bi_paths
                    }, cache_file)

            # 3. 计算相似度 (Batch Matrix Multiplication)
            # Shape: [n_prompts, n_images_in_batch]
            # [Fix] Ensure dtype consistency
            normalized_im_vectors = normalized_im_vectors.to(normalized_text_vectors.dtype)
            sim_matrix = torch.matmul(normalized_text_vectors, normalized_im_vectors.T)

            # 收集结果 (转为 CPU numpy)
            all_scores.append(sim_matrix.cpu().numpy())
            all_paths.extend(final_bi_paths)

    if not all_scores:
        return [], []

    # 4. 全局 Top-K 排序
    # 拼接所有 batch 的分数 -> Shape: [n_prompts, total_images]
    full_scores = np.concatenate(all_scores, axis=1)

    # 这里假设我们只针对第 0 个 prompt 进行检索 (通常 RAG 也是针对单句)
    # 如果需要支持多 prompt，可以在外层循环调用
    single_prompt_scores = full_scores[0]

    # 获取 Top K 索引
    k = min(k, len(single_prompt_scores))
    top_indices = np.argsort(single_prompt_scores)[-k:][::-1] # 降序

    top_paths = [all_paths[i] for i in top_indices]
    top_scores = single_prompt_scores[top_indices].tolist()

    return top_paths, top_scores


def get_siglip_similarities(prompts, image_paths, embeddings_path="", bs=1024, k=50, device='cuda:0', save=False, cache_dir=None):
    """
    使用 SigLIP (ViT-SO400M) 计算文本与图像库的相似度。
    SigLIP 通常比 CLIP 对细节理解更好。
    """
    # 加载模型
    try:
        model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384', cache_dir=cache_dir, device=device)
        tokenizer = get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP-384', cache_dir=cache_dir)
    except Exception as e:
        print(f"Error loading SigLIP: {e}")
        return [], []

    if isinstance(prompts, str):
        prompts = [prompts]

    text = tokenizer(prompts, context_length=model.context_length).to(device)

    all_scores = []
    all_paths = []

    with torch.no_grad():
        text_features = model.encode_text(text)
        normalized_text_vectors = F.normalize(text_features, dim=-1)

        for bi in range(0, len(image_paths), bs):

            cache_file = os.path.join(embeddings_path, f"siglip_embeddings_b{bi}.pt")
            current_batch_paths = image_paths[bi : bi + bs]

            if os.path.exists(cache_file):
                try:
                    data = torch.load(cache_file, map_location=device)
                    cached_emb = data['normalized_siglip_embeddings']
                    # Dimension Check
                    if cached_emb.shape[-1] == normalized_text_vectors.shape[-1]:
                        normalized_im_vectors = cached_emb
                        final_bi_paths = data.get('paths', current_batch_paths)
                    else:
                        print(f"Warning: Cached SigLIP embeddings in {cache_file} have dimension {cached_emb.shape[-1]}, but model has {normalized_text_vectors.shape[-1]}. Ignoring cache.")
                        normalized_im_vectors = None
                except Exception as e:
                    print(f"Error loading cache {cache_file}: {e}")
                    normalized_im_vectors = None

            if not os.path.exists(cache_file) or normalized_im_vectors is None:
                # 仅当 save=True 时才计算并保存，否则跳过 (为了速度)
                images = []
                valid_paths = []
                for path in current_batch_paths:
                    try:
                        img_pil = Image.open(path).convert("RGB")
                        image = preprocess(img_pil).unsqueeze(0).to(device)
                        images.append(image)
                        valid_paths.append(path)
                    except:
                        continue

                if not images: continue

                images = torch.stack(images).squeeze(1)
                image_features = model.encode_image(images)
                normalized_im_vectors = F.normalize(image_features, dim=-1)
                final_bi_paths = valid_paths

                if embeddings_path:
                    os.makedirs(embeddings_path, exist_ok=True)
                    torch.save({
                        "normalized_siglip_embeddings": normalized_im_vectors,
                        "paths": final_bi_paths
                    }, cache_file)
            else:
                # 如果没有缓存且不强制保存，跳过此 batch
                continue

            # [Fix] Ensure dtype consistency
            normalized_im_vectors = normalized_im_vectors.to(normalized_text_vectors.dtype)
            sim_matrix = torch.matmul(normalized_text_vectors, normalized_im_vectors.T)
            all_scores.append(sim_matrix.cpu().numpy())
            all_paths.extend(final_bi_paths)

    if not all_scores:
        return [], []

    full_scores = np.concatenate(all_scores, axis=1)
    single_prompt_scores = full_scores[0]

    k = min(k, len(single_prompt_scores))
    top_indices = np.argsort(single_prompt_scores)[-k:][::-1]

    top_paths = [all_paths[i] for i in top_indices]
    top_scores = single_prompt_scores[top_indices].tolist()

    return top_paths, top_scores


def get_siglip2_similarities(prompts, image_paths, embeddings_path="", bs=1024, k=50, device='cuda:0', save=False):
    """
    SigLIP2 Retrieval (google/siglip2-base-patch16-224)
    """
    try:
        from transformers import AutoModel, AutoProcessor
        model_name = "google/siglip2-base-patch16-224"
        # Use flash_attention_2 if available for efficiency, though not strictly required
        # The user doc mentions attn_implementation="flash_attention_2"
        model = AutoModel.from_pretrained(model_name, attn_implementation="sdpa").to(device).eval()
        processor = AutoProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading SigLIP2: {e}")
        return [], []

    if isinstance(prompts, str):
        prompts = [prompts]

    # Process Text
    # Note: SigLIP2 expects padding="max_length" and max_length=64
    try:
        text_inputs = processor(text=prompts, padding="max_length", max_length=64, return_tensors="pt").to(device)
    except Exception as e:
        print(f"Error processing text for SigLIP2: {e}")
        return [], []

    all_scores = []
    all_paths = []

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        normalized_text_vectors = F.normalize(text_features, dim=-1)

        for bi in range(0, len(image_paths), bs):
            cache_file = os.path.join(embeddings_path, f"siglip2_embeddings_b{bi}.pt")
            current_batch_paths = image_paths[bi : bi + bs]

            if os.path.exists(cache_file):
                try:
                    data = torch.load(cache_file, map_location=device, weights_only=False)
                    cached_emb = data['normalized_siglip2_embeddings']
                     # Dimension Check
                    if cached_emb.shape[-1] == normalized_text_vectors.shape[-1]:
                        normalized_im_vectors = cached_emb
                        final_bi_paths = data.get('paths', current_batch_paths)
                    else:
                        print(f"Warning: Cached SigLIP2 embeddings in {cache_file} have dimension {cached_emb.shape[-1]}, but model has {normalized_text_vectors.shape[-1]}. Ignoring cache.")
                        normalized_im_vectors = None

                except Exception as e:
                    print(f"Error loading cache {cache_file}: {e}")
                    normalized_im_vectors = None

            if not os.path.exists(cache_file) or normalized_im_vectors is None:
                if not save: continue

                images = []
                valid_paths = []
                for path in current_batch_paths:
                    try:
                        img_pil = Image.open(path).convert("RGB")
                        images.append(img_pil)
                        valid_paths.append(path)
                    except: continue

                if not images: continue

                # Process Images
                image_inputs = processor(images=images, return_tensors="pt").to(device)
                image_features = model.get_image_features(**image_inputs)
                normalized_im_vectors = F.normalize(image_features, dim=-1)
                final_bi_paths = valid_paths

                if embeddings_path:
                    os.makedirs(embeddings_path, exist_ok=True)
                    torch.save({
                        "normalized_siglip2_embeddings": normalized_im_vectors,
                        "paths": final_bi_paths
                    }, cache_file)

            # Ensure dtype consistency
            normalized_im_vectors = normalized_im_vectors.to(normalized_text_vectors.dtype)
            sim_matrix = torch.matmul(normalized_text_vectors, normalized_im_vectors.T)
            all_scores.append(sim_matrix.cpu().numpy())
            all_paths.extend(final_bi_paths)

    if not all_scores:
        return [], []

    full_scores = np.concatenate(all_scores, axis=1)
    single_prompt_scores = full_scores[0]

    k = min(k, len(single_prompt_scores))
    top_indices = np.argsort(single_prompt_scores)[-k:][::-1]

    top_paths = [all_paths[i] for i in top_indices]
    top_scores = single_prompt_scores[top_indices].tolist()

    return top_paths, top_scores


def retrieve_img_per_caption(captions, image_paths, embeddings_path="", k=3, device='cuda', method='CLIP', adapter_path=None, use_hybrid=False, external_model=None, external_processor=None):
    """
    统一入口函数。
    返回:
      - List[List[str]] (每个 caption 对应的 k 个图片路径)
      - List[List[float]] (每个 caption 对应的 k 个相似度分数)
    """
    all_retrieved_paths = []
    all_retrieved_scores = []

    # Check for global shared model if not provided
    if external_model is None:
        import sys
        external_model = getattr(sys.modules.get("__main__"), "GLOBAL_QWEN_MODEL", None)
        external_processor = getattr(sys.modules.get("__main__"), "GLOBAL_QWEN_PROCESSOR", None)

    for caption in captions:
        if method == 'CLIP':
            paths, scores = get_clip_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device
            )
        elif method == 'SigLIP':
            paths, scores = get_siglip_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device, save=True
            )
        elif method == 'SigLIP2':
            paths, scores = get_siglip2_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device, save=True
            )
        elif method == 'LongCLIP':
            paths, scores = get_longclip_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device
            )
        elif method == 'Qwen3-VL':
            paths, scores = get_qwen3_vl_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device, save=True,
                adapter_path=adapter_path,
                external_model=external_model,
                external_processor=external_processor
            )
        else:
            print(f"Unknown method {method}, falling back to CLIP")
            paths, scores = get_clip_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device
            )

        all_retrieved_paths.append(paths)
        all_retrieved_scores.append(scores)

    return all_retrieved_paths, all_retrieved_scores
def get_longclip_similarities(prompts, image_paths, embeddings_path="", bs=1024, k=50, device='cuda:0'):
    """
    Long-CLIP Retrieval Logic
    """
    if not LONGCLIP_AVAILABLE:
        print("Long-CLIP not available.")
        return [], []

    # Load Long-CLIP Model
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Long-CLIP/checkpoints/longclip-L.pt')
    if not os.path.exists(ckpt_path):
        print(f"Long-CLIP checkpoint not found at {ckpt_path}")
        return [], []

    model, preprocess = longclip.load(ckpt_path, device=device)
    model.eval()

    if isinstance(prompts, str):
        prompts = [prompts]

    text_features_list = []
    with torch.no_grad():
        for p in prompts:
            # Long-CLIP supports up to 248 tokens. We truncate if longer.
            text = longclip.tokenize([p], truncate=True).to(device)
            feat = model.encode_text(text)
            text_features_list.append(feat)

        text_features = torch.cat(text_features_list, dim=0)
        normalized_text_vectors = F.normalize(text_features, p=2, dim=1)

        all_scores = []
        all_paths = []

        for bi in range(0, len(image_paths), bs):
            cache_file = os.path.join(embeddings_path, f"longclip_embeddings_b{bi}.pt")
            current_batch_paths = image_paths[bi : bi + bs]

            if os.path.exists(cache_file):
                try:
                    data = torch.load(cache_file, map_location=device, weights_only=False)
                    cached_emb = data['normalized_clip_embeddings']
                    # Dimension Check
                    if cached_emb.shape[-1] == normalized_text_vectors.shape[-1]:
                        normalized_im_vectors = cached_emb
                        final_bi_paths = data.get('paths', current_batch_paths)
                    else:
                        print(f"Warning: Cached Long-CLIP embeddings in {cache_file} have dimension {cached_emb.shape[-1]}, but model has {normalized_text_vectors.shape[-1]}. Ignoring cache.")
                        normalized_im_vectors = None
                except Exception as e:
                    print(f"Error loading cache {cache_file}: {e}")
                    normalized_im_vectors = None

            if not os.path.exists(cache_file) or normalized_im_vectors is None:
                images = []
                valid_paths = []
                for path in current_batch_paths:
                    try:
                        img_pil = Image.open(path).convert("RGB")
                        image = preprocess(img_pil).unsqueeze(0).to(device)
                        images.append(image)
                        valid_paths.append(path)
                    except Exception as e:
                        print(f"Warning: Could not read {path}: {e}")
                        continue

                if not images: continue

                images = torch.stack(images).squeeze(1)
                image_features = model.encode_image(images)
                normalized_im_vectors = F.normalize(image_features, p=2, dim=1)
                final_bi_paths = valid_paths

                if embeddings_path:
                    os.makedirs(embeddings_path, exist_ok=True)
                    torch.save({
                        "normalized_clip_embeddings": normalized_im_vectors,
                        "paths": final_bi_paths
                    }, cache_file)

            normalized_im_vectors = normalized_im_vectors.to(normalized_text_vectors.dtype)
            sim_matrix = torch.matmul(normalized_text_vectors, normalized_im_vectors.T)
            all_scores.append(sim_matrix.cpu().numpy())
            all_paths.extend(final_bi_paths)

    if not all_scores:
        return [], []

    full_scores = np.concatenate(all_scores, axis=1)
    single_prompt_scores = full_scores[0]

    k = min(k, len(single_prompt_scores))
    top_indices = np.argsort(single_prompt_scores)[-k:][::-1]

    top_paths = [all_paths[i] for i in top_indices]
    top_scores = single_prompt_scores[top_indices].tolist()

    return top_paths, top_scores

def get_qwen3_vl_similarities(prompts, image_paths, embeddings_path="", bs=1, k=50, device='cuda:0', save=False, adapter_path=None, external_model=None, external_processor=None):
    """
    Qwen3-VL Retrieval
    """
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq

        if external_model is not None and external_processor is not None:
            print("Using shared external Qwen3-VL model for retrieval...")
            model = external_model
            processor = external_processor
        else:
            # Use local path
            model_name = "/home/tingyu/imageRAG/Qwen3-VL-4B-Instruct"

            print(f"Loading Qwen3-VL from {model_name}...")
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True
            )

            if adapter_path:
                try:
                    from peft import PeftModel
                    print(f"Loading LoRA adapter from {adapter_path}...")
                    model = PeftModel.from_pretrained(model, adapter_path)
                except ImportError:
                    print("Warning: peft not installed, cannot load adapter.")
                except Exception as e:
                    print(f"Error loading adapter: {e}")

            model.eval()
    except Exception as e:
        print(f"Error loading Qwen3-VL: {e}")
        return [], []

    if isinstance(prompts, str):
        prompts = [prompts]

    # 1. Encode Queries (Text)
    query_embeddings = []
    with torch.no_grad():
        for p in prompts:
            # Instruction for retrieval
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Retrieve the image that matches the description: {p}"}
                    ]
                }
            ]
            # Prepare inputs
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            # Forward pass
            outputs = model(**inputs, output_hidden_states=True)
            # Last layer, last token
            last_hidden_state = outputs.hidden_states[-1]
            emb = last_hidden_state[:, -1, :] # [1, Dim]
            query_embeddings.append(emb)

    query_embeddings = torch.cat(query_embeddings, dim=0)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)

    all_scores = []
    all_paths = []

    # 2. Encode Documents (Images)
    print(f"Encoding {len(image_paths)} images with Qwen3-VL (Batch Size: {bs})...")
    with torch.no_grad():
        for bi in tqdm(range(0, len(image_paths), bs), desc="Qwen3-VL Embedding"):
            cache_file = os.path.join(embeddings_path, f"qwen3_vl_embeddings_b{bi}.pt")
            current_batch_paths = image_paths[bi : bi + bs]

            batch_doc_embeddings = None
            final_bi_paths = []

            if os.path.exists(cache_file):
                try:
                    data = torch.load(cache_file, map_location=device, weights_only=False)
                    cached_emb = data['embeddings']
                    # Dimension Check
                    if cached_emb.shape[-1] == query_embeddings.shape[-1]:
                        batch_doc_embeddings = cached_emb
                        final_bi_paths = data.get('paths', current_batch_paths)
                    else:
                        print(f"Warning: Cached embeddings in {cache_file} have dimension {cached_emb.shape[-1]}, but model has {query_embeddings.shape[-1]}. Ignoring cache.")
                except Exception as e:
                    print(f"Error loading cache {cache_file}: {e}")

            if batch_doc_embeddings is None and save:
                images = []
                valid_paths = []
                for path in current_batch_paths:
                    try:
                        images.append(Image.open(path).convert("RGB"))
                        valid_paths.append(path)
                    except: continue

                if not images: continue

                batch_embs = []
                for img in images:
                    # Construct message with image
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": "Represent this image for retrieval."}
                            ]
                        }
                    ]
                    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = processor(
                        text=[text_prompt],
                        images=[img],
                        return_tensors="pt",
                        padding=True
                    ).to(device)

                    outputs = model(**inputs, output_hidden_states=True)
                    last_hidden_state = outputs.hidden_states[-1]
                    emb = last_hidden_state[:, -1, :]
                    batch_embs.append(emb)

                if batch_embs:
                    batch_doc_embeddings = torch.cat(batch_embs, dim=0)
                    final_bi_paths = valid_paths

                    if embeddings_path:
                        os.makedirs(embeddings_path, exist_ok=True)
                        torch.save({"embeddings": batch_doc_embeddings, "paths": final_bi_paths}, cache_file)

            if batch_doc_embeddings is not None:
                batch_doc_embeddings = batch_doc_embeddings.to(device)
                batch_doc_embeddings = F.normalize(batch_doc_embeddings, p=2, dim=-1)

                sim_matrix = query_embeddings @ batch_doc_embeddings.T
                all_scores.append(sim_matrix.float().cpu().numpy())
                all_paths.extend(final_bi_paths)

    if not all_scores:
        return [], []

    full_scores = np.concatenate(all_scores, axis=1)

    final_paths_list = []
    final_scores_list = []

    for i in range(len(prompts)):
        scores = full_scores[i]
        top_indices = np.argsort(scores)[-k:][::-1]
        top_paths = [all_paths[idx] for idx in top_indices]
        top_scores = scores[top_indices].tolist()
        final_paths_list.append(top_paths)
        final_scores_list.append(top_scores)

    del model
    torch.cuda.empty_cache()

    return final_paths_list[0], final_scores_list[0]
