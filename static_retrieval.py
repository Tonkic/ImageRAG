import os
import torch
import clip
from open_clip import create_model_from_pretrained, get_tokenizer
import torch.nn.functional as F
from PIL import Image
import numpy as np

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
                data = torch.load(cache_file, map_location=device)
                normalized_im_vectors = data['normalized_clip_embeddings']
                # 校验缓存路径是否匹配 (可选)
                final_bi_paths = data.get('paths', current_batch_paths)
            else:
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
                data = torch.load(cache_file, map_location=device)
                normalized_im_vectors = data['normalized_siglip_embeddings']
                final_bi_paths = data.get('paths', current_batch_paths)

            elif save:
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


def retrieve_img_per_caption(captions, image_paths, embeddings_path="", k=3, device='cuda', method='CLIP'):
    """
    统一入口函数。
    返回:
      - List[List[str]] (每个 caption 对应的 k 个图片路径)
      - List[List[float]] (每个 caption 对应的 k 个相似度分数)
    """
    all_retrieved_paths = []
    all_retrieved_scores = []

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