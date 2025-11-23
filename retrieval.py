import os
import torch
import clip
from open_clip import create_model_from_pretrained, get_tokenizer
import torch.nn.functional as F
from PIL import Image
import numpy as np

# (为简洁起见，省略了 message_gpt, rerank_BM25, gpt_rerank, retrieve_from_small_set)

def get_clip_similarities(prompts, image_paths, embeddings_path="", bs=1024, k=50, device='cuda:1'):
    """
    修改：
    1. 优化了循环外的 Top-K 查找。
    2. 返回 (paths, scores, embeddings)。
    """
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 确保 prompts 是一个列表
    if isinstance(prompts, str):
        prompts = [prompts]

    text = clip.tokenize(prompts).to(device)

    all_paths = []
    all_scores_list = []
    all_embeddings_list = []

    with torch.no_grad():
        text_features = model.encode_text(text)
        normalized_text_vectors = torch.nn.functional.normalize(text_features, p=2, dim=1)

        if bs >= len(image_paths):
            bs = len(image_paths)
            end = len(image_paths)
        else:
            end = len(image_paths) - bs

        for bi in range(0, end, bs):
            if os.path.exists(os.path.join(embeddings_path, f"clip_embeddings_b{bi}.pt")):
                normalized_ims = torch.load(os.path.join(embeddings_path, f"clip_embeddings_b{bi}.pt"), map_location=device)
                normalized_im_vectors = normalized_ims['normalized_clip_embeddings']
                final_bi_paths = normalized_ims['paths']
            else:
                print(f"Warning: Missing embedding file clip_embeddings_b{bi}.pt. 正在跳过 batch {bi}。")
                print("请在运行 smart_rag_dispatcher.py 之前先生成 embedding 缓存。")
                continue # (跳过这个 batch)

            # (在 GPU 上计算相似度)
            text_similarity_matrix = torch.matmul(normalized_text_vectors, normalized_im_vectors.T)

            # (收集所有结果)
            all_paths.extend(final_bi_paths)
            all_scores_list.append(text_similarity_matrix.cpu()) # (稍后处理)
            all_embeddings_list.append(normalized_im_vectors) # (保留在 GPU 上)

    if not all_scores_list:
        return np.array([]), np.array([]), torch.tensor([])

    # (在循环外处理所有分数)
    all_scores = torch.cat(all_scores_list, dim=1).numpy().squeeze()
    all_embeddings = torch.cat(all_embeddings_list, dim=0)

    # (确保 all_scores 至少是一维的)
    all_scores = np.atleast_1d(all_scores)

    # (我们只处理第一个 prompt 的情况，因为 VLM features 列表通常只用第一个)
    if all_scores.ndim > 1:
        all_scores = all_scores[0]

    top_indices = all_scores.argsort()[-k:]

    # (确保 all_paths 是 numpy 数组以便索引)
    all_paths = np.array(all_paths)

    top_text_im_paths = all_paths[top_indices]
    top_text_im_scores = all_scores[top_indices]
    top_img_embeddings = all_embeddings[top_indices]

    # (按降序返回)
    return top_text_im_paths[::-1], top_text_im_scores[::-1], top_img_embeddings.flip(dims=[0])


def get_siglip_similarities(prompts, image_paths, embeddings_path="", bs=1024, k=50, device='cuda:2', save=False, cache_dir=None):
    """
    修改：
    1. 优化了循环外的 Top-K 查找。
    2. 返回 (paths, scores, embeddings)。
    3. 修复了 'save=True' 逻辑，以便在 embedding 不存在时创建它们。
    """
    model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384', cache_dir=cache_dir, device=device)
    tokenizer = get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP-384', cache_dir=cache_dir)

    if isinstance(prompts, str):
        prompts = [prompts]

    text = tokenizer(prompts, context_length=model.context_length).to(device)

    all_paths = []
    all_scores_list = []
    all_embeddings_list = []

    with torch.no_grad():
        text_features = model.encode_text(text)
        normalized_text_vectors = F.normalize(text_features, dim=-1)

        if bs >= len(image_paths):
            bs = len(image_paths)
            end = len(image_paths)
        else:
            end = len(image_paths) - bs

        for bi in range(0, end, bs):
            if os.path.exists(os.path.join(embeddings_path, f"siglip_embeddings_b{bi}.pt")):
                normalized_ims = torch.load(os.path.join(embeddings_path, f"siglip_embeddings_b{bi}.pt"), map_location=device)
                normalized_im_vectors = normalized_ims['normalized_siglip_embeddings']#.to(device)
                final_bi_paths = normalized_ims['paths']

            elif save: # (如果 save=True，则动态创建 embedding)
                to_remove = []
                images = []
                for i in range(bs):
                    try:
                        image = preprocess(Image.open(image_paths[bi+i])).unsqueeze(0).to(device)
                        images.append(image)
                    except:
                        print(f"couldn't read {image_paths[bi+i]}")
                        to_remove.append(image_paths[bi+i])
                        continue
                if not images: continue
                images = torch.stack(images).squeeze(1).to(device)
                image_features = model.encode_image(images)
                normalized_im_vectors = F.normalize(image_features, dim=-1)
                final_bi_paths = [path for path in image_paths[bi:bi+bs] if path not in to_remove]
                if embeddings_path != "" and save:
                    os.makedirs(embeddings_path, exist_ok=True)
                    torch.save({"normalized_siglip_embeddings": normalized_im_vectors, "paths": final_bi_paths},
                               os.path.join(embeddings_path, f"siglip_embeddings_b{bi}.pt"))
            else:
                print(f"Warning: Missing embedding file siglip_embeddings_b{bi}.pt and save=False. Skipping batch {bi}.")
                continue

            text_similarity_matrix = torch.matmul(normalized_text_vectors, normalized_im_vectors.T)

            all_paths.extend(final_bi_paths)
            all_scores_list.append(text_similarity_matrix.cpu())
            all_embeddings_list.append(normalized_im_vectors)

    if not all_scores_list:
        return np.array([]), np.array([]), torch.tensor([])

    all_scores = torch.cat(all_scores_list, dim=1).numpy().squeeze()
    all_embeddings = torch.cat(all_embeddings_list, dim=0)
    all_scores = np.atleast_1d(all_scores)

    if all_scores.ndim > 1:
        all_scores = all_scores[0]

    top_indices = all_scores.argsort()[-k:]
    all_paths = np.array(all_paths)

    top_text_im_paths = all_paths[top_indices]
    top_text_im_scores = all_scores[top_indices]
    top_img_embeddings = all_embeddings[top_indices]

    return top_text_im_paths[::-1], top_text_im_scores[::-1], top_img_embeddings.flip(dims=[0])


# --------------------------------------------------
# --- ！！！关键修改：retrieve_img_per_caption！！！ ---
# --------------------------------------------------
def retrieve_img_per_caption(captions, image_paths, embeddings_path="", k=3, device='cuda', method='CLIP'):
    """
    修改：现在返回 (all_paths, all_scores, all_embeddings)
    """
    all_paths = []
    all_scores = []
    all_embeddings = []

    model_func = get_clip_similarities
    if method == 'SigLIP':
        model_func = get_siglip_similarities
    elif method == 'gpt_rerank':
        print("Warning: 'gpt_rerank' is not compatible with reranking. Falling back to CLIP.")
        method = 'CLIP'
        model_func = get_clip_similarities
    elif method == 'MoE':
        print("Warning: 'MoE' is not compatible with reranking. Falling back to CLIP.")
        method = 'CLIP'
        model_func = get_clip_similarities

    # (smart_rag_dispatcher.py 中的 CLIP_Rerank 逻辑在调用此函数 *之前* 处理,
    # 此处 method 将是 'CLIP' 或 'SigLIP')

    for caption in captions:

        # ！！！关键修复：动态设置 kwargs！！！
        kwargs = {
            "prompts": [caption],
            "image_paths": image_paths,
            "embeddings_path": embeddings_path,
            "bs": min(2048, len(image_paths)),
            "k": k,
            "device": device
        }

        # 只有 SigLIP 才需要 'save' 参数
        if method == 'SigLIP':
            kwargs['save'] = True

        paths, scores, embeddings = model_func(**kwargs)

        print(f"Method {method} pairs:", (paths[:5], scores[:5])) # 只打印前5个
        all_paths.append(paths)
        all_scores.append(scores)
        all_embeddings.append(embeddings)

    return all_paths, all_scores, all_embeddings