import os
import torch
import clip
from open_clip import create_model_from_pretrained, get_tokenizer
import torch.nn.functional as F
from PIL import Image
import numpy as np

# -----------------------------------------------------------------------------
# Memory-Guided Retrieval Module
# -----------------------------------------------------------------------------

def get_clip_similarities(prompts, image_paths, embeddings_path="", bs=1024, k=50, device='cuda:0'):
    """
    CLIP 检索核心逻辑
    """
    model, preprocess = clip.load("ViT-B/32", device=device)

    if isinstance(prompts, str):
        prompts = [prompts]

    # [修改] 增加 truncate=True 参数，自动截断超过 77 token 的文本
    # 这样可以保留尽可能多的有效信息，同时避免 RuntimeError
    text = clip.tokenize(prompts, truncate=True).to(device)

    all_scores = []
    all_paths = []

    with torch.no_grad():
        text_features = model.encode_text(text)
        normalized_text_vectors = F.normalize(text_features, p=2, dim=1)

        for bi in range(0, len(image_paths), bs):
            cache_file = os.path.join(embeddings_path, f"clip_embeddings_b{bi}.pt")
            current_batch_paths = image_paths[bi : bi + bs]

            if os.path.exists(cache_file):
                data = torch.load(cache_file, map_location=device)
                normalized_im_vectors = data['normalized_clip_embeddings']
                final_bi_paths = data.get('paths', current_batch_paths)
            else:
                images = []
                valid_paths = []
                for path in current_batch_paths:
                    try:
                        # 强制转 RGB
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

            # [Fix] Ensure dtype consistency (e.g., if cache is float16 but text is float32 on CPU)
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


def get_siglip_similarities(prompts, image_paths, embeddings_path="", bs=1024, k=50, device='cuda:0', save=False, cache_dir=None):
    """
    使用 SigLIP (ViT-SO400M) 计算文本与图像库的相似度。
    """
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
    统一检索入口。
    支持 method: 'CLIP', 'SigLIP', 'Hybrid' (RRF Fusion)
    """
    all_retrieved_paths = []
    all_retrieved_scores = []

    for caption in captions:
        if method == 'Hybrid':
            # 1. Get CLIP Results (Top-K * 2 to ensure overlap)
            k_expanded = k * 2
            paths_c, scores_c = get_clip_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k_expanded, device=device
            )

            # 2. Get SigLIP Results
            paths_s, scores_s = get_siglip_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k_expanded, device=device, save=True
            )

            # 3. Reciprocal Rank Fusion (RRF)
            # Score = 1 / (k + rank)
            rrf_scores = {}

            # Process CLIP
            for rank, path in enumerate(paths_c):
                if path not in rrf_scores: rrf_scores[path] = 0.0
                rrf_scores[path] += 1.0 / (60 + rank)

            # Process SigLIP
            for rank, path in enumerate(paths_s):
                if path not in rrf_scores: rrf_scores[path] = 0.0
                rrf_scores[path] += 1.0 / (60 + rank)

            # Sort by RRF score
            sorted_paths = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
            sorted_paths = sorted_paths[:k]

            # Normalize scores for compatibility (just use RRF score or max of original?)
            # For simplicity, we return the RRF score, but scaled to look like a similarity (0-1)
            # Max RRF is approx 1/60 + 1/60 = 0.033. Let's scale by 30 to get ~1.0
            final_scores = [rrf_scores[p] * 30.0 for p in sorted_paths]

            all_retrieved_paths.append(sorted_paths)
            all_retrieved_scores.append(final_scores)

        elif method == 'CLIP':
            paths, scores = get_clip_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device
            )
            all_retrieved_paths.append(paths)
            all_retrieved_scores.append(scores)

        elif method == 'SigLIP':
            paths, scores = get_siglip_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device, save=True
            )
            all_retrieved_paths.append(paths)
            all_retrieved_scores.append(scores)

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