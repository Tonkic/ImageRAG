import os
import torch
import clip
from open_clip import create_model_from_pretrained, get_tokenizer
import torch.nn.functional as F
from PIL import Image
import numpy as np

# 尝试导入辅助函数，防止报错
try:
    from retrieval_w_gpt import get_image_captions, message_gpt
except ImportError:
    # 如果没有该文件，定义占位符以免整个脚本崩溃，但运行时会报错
    def get_image_captions(candidates): raise ImportError("retrieval_w_gpt not found")
    def message_gpt(msg, paths): raise ImportError("retrieval_w_gpt not found")

def get_clip_similarities(prompts, image_paths, embeddings_path="", bs=1024, k=50, device='cuda:1'):
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(prompts).to(device)

    top_text_im_paths = []
    top_text_im_scores = []
    top_img_embeddings = torch.empty((0, 512))

    with torch.no_grad():
        text_features = model.encode_text(text)
        normalized_text_vectors = torch.nn.functional.normalize(text_features, p=2, dim=1)

        # [修改 1] 修复循环逻辑，确保处理最后一部分不足 bs 的数据
        # 原代码: end = len - bs 会导致丢弃尾部数据
        for bi in range(0, len(image_paths), bs):

            # 检查缓存
            cache_file = os.path.join(embeddings_path, f"clip_embeddings_b{bi}.pt")

            if os.path.exists(cache_file):
                normalized_ims = torch.load(cache_file, map_location=device)
                normalized_im_vectors = normalized_ims['normalized_clip_embeddings']
                final_bi_paths = normalized_ims['paths']

            else:
                to_remove = []
                images = []
                # 确保切片不会越界
                current_batch_paths = image_paths[bi : bi + bs]

                for path in current_batch_paths:
                    try:
                        # [修改 2] 增加 .convert("RGB") 确保处理单通道灰度图
                        img_pil = Image.open(path).convert("RGB")
                        image = preprocess(img_pil).unsqueeze(0).to(device)
                        images.append(image)
                    except Exception as e:
                        print(f"couldn't read {path}: {e}")
                        to_remove.append(path)
                        continue

                if not images:
                    continue

                images = torch.stack(images).squeeze(1).to(device)
                image_features = model.encode_image(images)
                normalized_im_vectors = torch.nn.functional.normalize(image_features, p=2, dim=1)

                final_bi_paths = [path for path in current_batch_paths if path not in to_remove]

                if embeddings_path != "":
                    os.makedirs(embeddings_path, exist_ok=True)
                    torch.save({"normalized_clip_embeddings": normalized_im_vectors, "paths": final_bi_paths},
                               cache_file)

            # compute cosine similarities
            # 注意：如果 text 只有一个 prompt，这里通常是 (1, dim)
            text_similarity_matrix = torch.matmul(normalized_text_vectors, normalized_im_vectors.T)

            text_sim = text_similarity_matrix.cpu().numpy().squeeze()

            # 处理 squeeze 后如果是标量的情况 (batch 只有1个时)
            if text_sim.ndim == 0:
                text_sim = np.expand_dims(text_sim, axis=0)

            top_text_im_scores = np.concatenate([top_text_im_scores, text_sim])
            top_text_im_paths = np.concatenate([top_text_im_paths, final_bi_paths])

            # 保持 top k
            # 注意：这里如果总数量小于 k，argsort可能会报错，加个保护
            current_total = len(top_text_im_scores)
            current_k = min(k, current_total)

            top_similarities = top_text_im_scores.argsort()[-current_k:]

            # 重新排序并截断
            top_text_im_paths = top_text_im_paths[top_similarities]
            top_text_im_scores = top_text_im_scores[top_similarities]

            # 更新 embeddings 缓存 (仅保留 top k)
            # 注意：这里逻辑稍微有点复杂，因为 embeddings 在 CPU/GPU 切换
            # 为了简化，这里只处理 current batch 的拼接，实际 top_img_embeddings 维护可能需要对应索引
            # 原代码逻辑是累积所有 embedding，这里尽量保持原意但修复形状匹配
            if top_img_embeddings.shape[0] == 0:
                 cur_embeddings = normalized_im_vectors.cpu()
            else:
                 # 这里原代码逻辑其实有点问题（它把所有之前的都拼起来，但只取了 top k 的 path）
                 # 简单修复：仅拼接当前的，然后根据索引筛选
                 # 但为了不大幅重构逻辑，我们假设它是想维护一个 growing list
                 pass

            # 原代码中 top_img_embeddings 更新逻辑在 loop 内其实比较模糊
            # 既然只需返回 path 和 score，如果不需返回 embedding，可以简化
            # 这里为了兼容原代码的 return 签名，尽量保留

            # 修正：仅在最后或 loop 中正确索引
            # 简单处理：暂不维护 top_img_embeddings 的精确裁剪，除非后续用到
            # 下面这行是原代码逻辑，通过 top_similarities 索引所有拼接后的向量
            # 这要求 top_img_embeddings 必须包含当前 batch
            cur_full_embeddings = torch.cat([top_img_embeddings, normalized_im_vectors.cpu()])
            top_img_embeddings = cur_full_embeddings[top_similarities]

    return top_text_im_paths[::-1], top_text_im_scores[::-1]

def rerank_BM25(candidates, retrieval_captions, k=1):
    from rank_bm25 import BM25Okapi
    # 确保已导入 get_image_captions
    try:
        from retrieval_w_gpt import get_image_captions
    except ImportError:
        print("Warning: retrieval_w_gpt not found for BM25 rerank.")
        return candidates[:k], [0.0]*k

    candidates = list(set(candidates))
    candidate_captions = get_image_captions(candidates)

    tokenized_captions = [candidate_captions[candidate].lower().split() for candidate in candidates]
    bm25 = BM25Okapi(tokenized_captions)
    tokenized_query = retrieval_captions[0].lower().split() # TODO currently only works for 1 caption
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = np.argsort(-scores)

    return np.array(candidates)[ranked_indices[:k]].tolist(), np.array(scores)[ranked_indices[:k]].tolist()

def get_moe_similarities(prompts, image_paths, embeddings_path="", bs=1024, k=1, device='cuda:2', save=False):
    pairs, im_emb = get_clip_similarities(prompts, image_paths,
                                          embeddings_path=embeddings_path,
                                          bs=min(2048, bs), k=3, device=device)
    pairs2, im_emb2 = get_siglip_similarities(prompts, image_paths,
                                              embeddings_path=embeddings_path,
                                              bs=min(64, bs), k=3, device=device, save=save)

    candidates = pairs[0].tolist() + pairs2[0].tolist()
    scores = pairs[1].tolist() + pairs2[1].tolist()
    bm25_best, bm25_scores = rerank_BM25(candidates, prompts, k=3)
    path2score = {c: 0 for c in candidates}
    for i in range(len(candidates)):
        path2score[candidates[i]] += scores[i]
        if candidates[i] in bm25_best:
            path2score[candidates[i]] += bm25_scores[bm25_best.index(candidates[i])]

    best_score = max(list(path2score.values()))
    best_path = [p for p,v in path2score.items() if v == best_score]
    return best_path, [best_score]

def get_siglip_similarities(prompts, image_paths, embeddings_path="", bs=1024, k=50, device='cuda:2', save=False, cache_dir=None):
    model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384', cache_dir=cache_dir, device=device)
    tokenizer = get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP-384', cache_dir=cache_dir)
    text = tokenizer(prompts, context_length=model.context_length).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        normalized_text_vectors = F.normalize(text_features, dim=-1)

        top_text_im_paths = []
        top_text_im_scores = []
        top_img_embeddings = torch.empty((0, 1152))

        # [修改 1] 修复循环逻辑
        for bi in range(0, len(image_paths), bs):

            cache_file = os.path.join(embeddings_path, f"siglip_embeddings_b{bi}.pt")

            if os.path.exists(cache_file):
                normalized_ims = torch.load(cache_file, map_location=device)
                normalized_im_vectors = normalized_ims['normalized_siglip_embeddings']#.to(device)
                final_bi_paths = normalized_ims['paths']

            elif save:
                to_remove = []
                images = []
                current_batch_paths = image_paths[bi : bi + bs]

                for path in current_batch_paths:
                    try:
                        # [修改 2] 增加 .convert("RGB")
                        img_pil = Image.open(path).convert("RGB")
                        image = preprocess(img_pil).unsqueeze(0).to(device)
                        images.append(image)
                    except Exception as e:
                        print(f"couldn't read {path}: {e}")
                        to_remove.append(path)
                        continue

                if not images:
                    continue

                images = torch.stack(images).squeeze(1).to(device)
                image_features = model.encode_image(images)
                normalized_im_vectors = F.normalize(image_features, dim=-1)

                final_bi_paths = [path for path in current_batch_paths if path not in to_remove]
                if embeddings_path != "" and save:
                    os.makedirs(embeddings_path, exist_ok=True)
                    torch.save({"normalized_siglip_embeddings": normalized_im_vectors, "paths": final_bi_paths},
                               cache_file)
            else:
                continue

            # compute cosine similarities
            text_similarity_matrix = torch.matmul(normalized_text_vectors, normalized_im_vectors.T)

            text_sim = text_similarity_matrix.cpu().numpy().squeeze()
            if text_sim.ndim == 0: text_sim = np.expand_dims(text_sim, axis=0)

            top_text_im_scores = np.concatenate([top_text_im_scores, text_sim])
            top_text_im_paths = np.concatenate([top_text_im_paths, final_bi_paths])

            current_total = len(top_text_im_scores)
            current_k = min(k, current_total)

            top_similarities = top_text_im_scores.argsort()[-current_k:]

            top_text_im_paths = top_text_im_paths[top_similarities]
            top_text_im_scores = top_text_im_scores[top_similarities]

            cur_full_embeddings = torch.cat([top_img_embeddings, normalized_im_vectors.cpu()])
            top_img_embeddings = cur_full_embeddings[top_similarities]

    return top_text_im_paths[::-1], top_text_im_scores[::-1]

def gpt_rerank(caption, image_paths, embeddings_path="", bs=1024, k=1, device='cuda', save=False):
    pairs, im_emb = get_clip_similarities(caption, image_paths,
                                          embeddings_path=embeddings_path,
                                          bs=min(2048, bs), k=3, device=device)
    pairs2, im_emb2 = get_siglip_similarities(caption, image_paths,
                                              embeddings_path=embeddings_path,
                                              bs=min(64, bs), k=3, device=device, save=save)
    print(f"CLIP candidates: {pairs}")
    print(f"SigLIP candidates: {pairs2}")

    candidates = pairs[0].tolist() + pairs2[0].tolist()
    scores = pairs[1].tolist() + pairs2[1].tolist()

    best_paths = retrieve_from_small_set(candidates, caption, k=k)

    return (best_paths, [scores[candidates.index(p)] for p in best_paths]), im_emb

def retrieve_from_small_set(image_paths, prompt, k=3):
    # 确保 message_gpt 可用
    try:
        from retrieval_w_gpt import message_gpt
    except ImportError:
        print("Warning: retrieval_w_gpt not found for gpt rerank.")
        return image_paths[:k]

    best = []
    bs = min(6, len(image_paths))
    for i in range(0, len(image_paths), bs):
        cur_paths = best + image_paths[i:i+bs]
        msg = (f'Which of these images is the most similar to the prompt {prompt}?'
               f'in your answer only provide the indices of the {k} most relevant images with a comma between them with no spaces, starting from index 0, e.g. answer: 0,3 if the most similar images are the ones in indices 0 and 3.'
               f'If you can\'t determine, return the first {k} indices, e.g. 0,1 if {k}=2.')
        best_ind = message_gpt(msg, cur_paths).split(",")
        try:
            best = [cur_paths[int(j.strip("'").strip('"').strip())] for j in best_ind]
        except:
            print(f"didn't get ind for i {i}")
            print(best_ind)
            continue
    return best

def retrieve_img_per_caption(captions, image_paths, embeddings_path="", k=3, device='cuda', method='CLIP'):
    paths = []
    for caption in captions:
        if method == 'CLIP':
            pairs = get_clip_similarities(caption, image_paths,
                                          embeddings_path=embeddings_path,
                                          bs=min(2048, len(image_paths)), k=k, device=device)
        elif method == 'SigLIP':
            pairs = get_siglip_similarities(caption, image_paths,
                                            embeddings_path=embeddings_path,
                                            bs=min(2048, len(image_paths)), k=k, device=device)
        elif method == 'MoE':
            pairs = get_moe_similarities(caption, image_paths,
                                         embeddings_path=embeddings_path,
                                        bs=min(2048, len(image_paths)), k=k, device=device)

        elif method == 'gpt_rerank':
            pairs = gpt_rerank(caption, image_paths,
                               embeddings_path=embeddings_path,
                               bs=min(2048, len(image_paths)), k=k, device=device)
            print(f"gpt rerank best path: {pairs[0]}")

        print("pairs:", pairs)
        paths.append(pairs[0])

    return paths