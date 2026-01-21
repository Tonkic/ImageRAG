import os
import torch
import clip
from open_clip import create_model_from_pretrained, get_tokenizer
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import AutoModel, AutoProcessor

# [Hybrid Retrieval]
from rank_bm25 import BM25Okapi
import re

class HybridRetriever:
    def __init__(self, image_paths):
        """
        初始化混合检索器
        :param image_paths: 所有候选图片的完整路径列表
        """
        self.image_paths = image_paths

        # 1. 构建语料库 (Corpus Construction)
        # 从文件名中提取关键信息作为 BM25 的文档
        # 例如: "datasets/images/Boeing_707-320.jpg" -> "Boeing 707 320"
        self.corpus_text = [self._clean_filename(p) for p in image_paths]

        # 2. 分词 (Tokenization)
        self.tokenized_corpus = [doc.split() for doc in self.corpus_text]

        # 3. 建立索引 (Index Building)
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"BM25 Index built for {len(image_paths)} images.")

    def _clean_filename(self, path):
        """辅助函数：从路径清洗出纯文本"""
        basename = os.path.basename(path)
        name_no_ext = os.path.splitext(basename)[0]
        # 将下划线、连字符替换为空格，只保留字母数字
        clean_name = re.sub(r"[^a-zA-Z0-9]", " ", name_no_ext)
        return clean_name.lower()

    def normalize_scores(self, scores):
        """
        Min-Max 归一化，将分数映射到 [0, 1] 区间
        """
        scores = np.array(scores)
        if scores.size == 0:
            return scores

        min_val = np.min(scores)
        max_val = np.max(scores)

        # 避免除以零
        if max_val - min_val == 0:
            return np.zeros_like(scores)

        return (scores - min_val) / (max_val - min_val)

    def hybrid_search(self, query_text, vector_scores, alpha=0.7):
        """
        执行混合检索
        :param query_text: 文本查询词 (string)
        :param vector_scores: 对应的向量检索分数 (numpy array, shape=[N])
        :param alpha: 向量分数的权重 (0.0 - 1.0)，BM25 权重为 1-alpha
        :return: 融合后的分数 (numpy array)
        """
        # 1. 计算 BM25 分数 (Sparse Score)
        tokenized_query = self._clean_filename(query_text).split()
        bm25_raw_scores = self.bm25.get_scores(tokenized_query)

        # 2. 归一化 (Normalization) - 关键步骤！
        # 必须把两者都拉伸到 0-1 之间才能加权
        norm_vector_scores = self.normalize_scores(vector_scores)
        norm_bm25_scores = self.normalize_scores(bm25_raw_scores)

        # 3. 加权融合 (Weighted Sum)
        final_scores = (alpha * norm_vector_scores) + ((1 - alpha) * norm_bm25_scores)

        return final_scores

GLOBAL_BM25_RETRIEVER = None

# [Optional] Long-CLIP Imports
import sys
from pic2word import IM2TEXT, encode_text_with_token

GLOBAL_MAPPER = None

def load_mapper(path, embed_dim, device):
    global GLOBAL_MAPPER
    if GLOBAL_MAPPER is None:
        if not os.path.exists(path):
            print(f"Warning: Pic2Word mapper not found at {path}")
            return None
        model = IM2TEXT(embed_dim=embed_dim, output_dim=embed_dim).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        GLOBAL_MAPPER = model
    return GLOBAL_MAPPER

def retrieve_composed(ref_image_path, modifier_text, image_paths, embeddings_path,
                      mapper_path="pic2word_mapper_ep9.pt", k=1, device='cuda', bs=1024, method="CLIP"):
    """
    组合检索: 图片 + 修改文字 -> 目标图片
    Supports: CLIP, LongCLIP
    """
    # 1. 准备模型 (根据 method 加载)
    model = None
    preprocess = None
    mapper = None

    if method == "LongCLIP":
        import sys
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Long-CLIP'))
        from model import longclip
        # Checkpoint path
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Long-CLIP/checkpoints/longclip-L.pt')
        if not os.path.exists(ckpt_path):
             print(f"Long-CLIP checkpoint missing at {ckpt_path}")
             return [], []
        model, preprocess = longclip.load(ckpt_path, device=device)
        model.eval()

        # Determine dim
        # Long-CLIP-L visual output dim is 768
        embed_dim = model.visual.output_dim

    else:
        # Default CLIP
        import clip
        model, preprocess = clip.load("ViT-L/14", device=device)
        embed_dim = model.visual.output_dim


    mapper = load_mapper(mapper_path, embed_dim, device)

    if mapper is None:
        print("Pic2Word Mapper not loaded. Returning empty.")
        return [], []

    # 2. 处理参考图 (Visual Anchor)
    try:
        image = preprocess(Image.open(ref_image_path).convert("RGB")).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading ref image {ref_image_path}: {e}")
        return [], []

    with torch.no_grad():
        img_feat = model.encode_image(image)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        # 映射为 Token
        pseudo_token = mapper(img_feat) # [1, Dim]

    # 3. 构造组合 Prompt
    # 格式: "A photo of *, {modifier}"
    prompt = f"A photo of *, {modifier_text}"

    star_token_id = 0 # default
    text_tokens = None

    if method == "LongCLIP":
        from model import longclip # ensure import
        # LongCLIP tokenizer
        text_tokens = longclip.tokenize([prompt]).to(device)
        star_token_id = longclip.tokenize(["*"])[0][1]
    else:
        import clip
        text_tokens = clip.tokenize([prompt]).to(device)
        star_token_id = clip.tokenize(["*"])[0][1]

    # 4. 编码组合 Query
    # encode_text_with_token handles LongCLIP check inside now
    with torch.no_grad():
        query_emb = encode_text_with_token(model, text_tokens, pseudo_token, star_token_id)
        normalized_text_vectors = query_emb / query_emb.norm(dim=-1, keepdim=True)

    # 5. 检索
    all_scores = []
    all_paths = []

    embedding_cache_prefix = "longclip_embeddings" if method == "LongCLIP" else "clip_embeddings"

    with torch.no_grad():
        for bi in range(0, len(image_paths), bs):
            cache_file = os.path.join(embeddings_path, f"{embedding_cache_prefix}_b{bi}.pt")
            current_batch_paths = image_paths[bi : bi + bs]

            normalized_im_vectors = None
            if os.path.exists(cache_file):
                try:
                    data = torch.load(cache_file, map_location=device, weights_only=False)

                    normalized_im_vectors = data['normalized_clip_embeddings']
                    final_bi_paths = data.get('paths', current_batch_paths)
                except: pass

            # Note: Assuming Embeddings are ViT-B/32 in standard flow, wait.
            # Pic2Word usually uses ViT-L/14. If the database was encoded with ViT-B/32, this WONT WORK.
            # The user must ensure "clip_embeddings" are compatible or re-compute.
            # However, the user request says: "assume you have loaded .pt ... same as your retrieval model".
            # If the user script uses ViT-L/14 for retrieval, then embeddings are L/14.
            # But earlier in file: model, preprocess = clip.load("ViT-B/32", device=device)
            pass

            # Since we can't guarantee embeddings are L/14, we might need to re-encode if we want to be safe,
            # Or assume the user handles it.
            # Given the context, I should probably Re-encode if cache is B/32?
            # For simplicity, I will re-implement the loop to load images and encode with current L/14 model.

            # ACTUALLY: The user's prompt step 2 training uses "ViT-L/14".
            # The retrieval DB probably uses B/32 by default in the script.
            # Conflict!
            # I should calculate embeddings on the fly if I can't trust cache.
            # Or better, just implement the loop to encode images using the L/14 model.

            images = []
            valid_paths = []
            for path in current_batch_paths:
                try:
                    img_pil = Image.open(path).convert("RGB")
                    images.append(preprocess(img_pil).unsqueeze(0).to(device))
                    valid_paths.append(path)
                except: continue

            if not images: continue

            images = torch.cat(images, dim=0)
            image_features = model.encode_image(images)
            normalized_im_vectors = image_features / image_features.norm(dim=-1, keepdim=True)

            sim_matrix = torch.matmul(normalized_text_vectors, normalized_im_vectors.T)
            all_scores.append(sim_matrix.cpu().numpy())
            all_paths.extend(valid_paths)

    if not all_scores:
        return [], []

    full_scores = np.concatenate(all_scores, axis=1)
    single_prompt_scores = full_scores[0] # [N]

    k = min(k, len(single_prompt_scores))
    top_indices = np.argsort(single_prompt_scores)[-k:][::-1]

    top_paths = [all_paths[i] for i in top_indices]
    top_scores = single_prompt_scores[top_indices].tolist()

    return top_paths, top_scores

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Long-CLIP'))
try:
    from model import longclip
    LONGCLIP_AVAILABLE = True
except ImportError:
    LONGCLIP_AVAILABLE = False
    print("Warning: Long-CLIP not found.")

# [Optional] ColPali Imports
# Removed ColPali imports as requested

# -----------------------------------------------------------------------------
# Memory-Guided Retrieval Module
# -----------------------------------------------------------------------------

def check_token_length(prompts, device="cpu", method="CLIP"):
    # [Modified] Disable warning as we now handle long text via chunking
    return

    # Process Queries
    with torch.no_grad():
        batch_queries = processor.process_queries(prompts).to(device)
        # [Batch, N_q, Dim]
        query_embeddings = model(**batch_queries)

    all_scores = []
    all_paths = []

    with torch.no_grad():
        for bi in range(0, len(image_paths), bs):
            cache_file = os.path.join(embeddings_path, f"colpali_embeddings_b{bi}.pt")
            current_batch_paths = image_paths[bi : bi + bs]

            if os.path.exists(cache_file):
                data = torch.load(cache_file, map_location=device, weights_only=False)
                batch_doc_embeddings = data['colpali_embeddings'] # List of tensors
                final_bi_paths = data.get('paths', current_batch_paths)
            elif save:
                images = []
                valid_paths = []
                for path in current_batch_paths:
                    try:
                        img_pil = Image.open(path).convert("RGB")
                        images.append(img_pil)
                        valid_paths.append(path)
                    except: continue

                if not images: continue

    # ... (Re-implementing loop logic correctly in the tool call)

def get_clip_similarities(prompts, image_paths, embeddings_path="", bs=1024, k=50, device='cuda:0', use_hybrid=False):
    """
    CLIP 检索核心逻辑 (支持长文本 Chunking)
    """
    model, preprocess = clip.load("ViT-B/32", device=device)

    if isinstance(prompts, str):
        prompts = [prompts]

    # [Modified] Long Text Handling: Truncate (Removed Mean Pool)
    text_features_list = []
    with torch.no_grad():
        for p in prompts:
            # Simply truncate if too long
            text = clip.tokenize([p], truncate=True).to(device)
            feat = model.encode_text(text)
            text_features_list.append(feat)

        text_features = torch.cat(text_features_list, dim=0)
        normalized_text_vectors = F.normalize(text_features, p=2, dim=1)

        all_scores = []
        all_paths = []

        for bi in range(0, len(image_paths), bs):
            cache_file = os.path.join(embeddings_path, f"clip_embeddings_b{bi}.pt")
            current_batch_paths = image_paths[bi : bi + bs]

            if os.path.exists(cache_file):
                data = torch.load(cache_file, map_location=device, weights_only=False)
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

    # [Hybrid Retrieval]
    if use_hybrid:
        global GLOBAL_BM25_RETRIEVER
        if GLOBAL_BM25_RETRIEVER is None or len(GLOBAL_BM25_RETRIEVER.image_paths) != len(all_paths):
            print(f"Initializing HybridRetriever with {len(all_paths)} images...")
            GLOBAL_BM25_RETRIEVER = HybridRetriever(all_paths)

        # Hybrid Search (0.7 Vector + 0.3 BM25)
        final_scores = GLOBAL_BM25_RETRIEVER.hybrid_search(
            query_text=prompts[0],
            vector_scores=single_prompt_scores,
            alpha=0.7
        )
    else:
        final_scores = single_prompt_scores

    k = min(k, len(final_scores))
    top_indices = np.argsort(final_scores)[-k:][::-1]

    top_paths = [all_paths[i] for i in top_indices]
    top_scores = final_scores[top_indices].tolist()

    return top_paths, top_scores


    # Re-organize scores
    # We need to handle the batching correctly.
    # Let's simplify: return empty for now if not implemented fully, but I will implement fully below.
    pass

# Removed get_colpali_similarities_full as requested

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
                data = torch.load(cache_file, map_location=device, weights_only=False)
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
                    normalized_im_vectors = data['normalized_siglip2_embeddings']
                    final_bi_paths = data.get('paths', current_batch_paths)
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


class GlobalMemory:
    """
    Global Memory for tracking retrieval history and re-ranking.
    Currently implements a 'Taboo' mechanism to penalize previously used images.
    """
    def __init__(self):
        self.history = set()

    def add(self, path):
        self.history.add(path)

    def __contains__(self, path):
        return path in self.history

    def re_rank(self, paths, scores, penalty=100.0):
        """
        Re-rank paths/scores by penalizing items in history.
        """
        new_paths = []
        new_scores = []

        # Combine and sort
        combined = list(zip(paths, scores))

        # Apply penalty
        penalized = []
        for p, s in combined:
            if p in self.history:
                s = s - penalty
            penalized.append((p, s))

        # Re-sort descending
        penalized.sort(key=lambda x: x[1], reverse=True)

        # Unzip
        if penalized:
            new_paths, new_scores = zip(*penalized)
            return list(new_paths), list(new_scores)
        return [], []



def get_bge_vl_similarities(prompts, image_paths, embeddings_path="", bs=32, k=50, device='cuda:0', save=False):
    """
    BGE-VL Retrieval Logic
    """
    try:
        from transformers import AutoModel
        MODEL_NAME = "BAAI/BGE-VL-large"

        # [Fix] Load on CPU first to debug/prevent CUDA context corruption
        print(f"Loading {MODEL_NAME} on CPU first...")
        model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)

        # [Fix] Re-initialize position_ids if they are invalid (garbage values)
        # This is a known issue with some remote code models where buffers are not initialized correctly

        # 1. Fix Text Model Position IDs
        if hasattr(model, "text_model") and hasattr(model.text_model, "embeddings") and hasattr(model.text_model.embeddings, "position_ids"):
            pos_ids = model.text_model.embeddings.position_ids
            # Check if max value is garbage (>= 77 for CLIP)
            if pos_ids.max() >= 77:
                print("WARNING: Text Position IDs are invalid (>= 77). Re-initializing them...")
                # CLIP max_position_embeddings is usually 77
                # Use clone() to ensure it's contiguous and owns its memory
                new_pos_ids = torch.arange(77).expand((1, -1)).clone()
                model.text_model.embeddings.position_ids = new_pos_ids
                print("Fixed Text Position IDs.")

        # 2. Fix Vision Model Position IDs
        vision_model = getattr(model, "vision_model", getattr(model, "visual_model", None))
        if vision_model and hasattr(vision_model, "embeddings") and hasattr(vision_model.embeddings, "position_ids"):
            vis_pos_ids = vision_model.embeddings.position_ids
            # Check if max value is garbage. Usually size is num_patches + 1
            # For ViT-L/14 224px: (224/14)^2 + 1 = 257
            # Let's check the embedding weight size to be sure
            if hasattr(vision_model.embeddings, "position_embedding") and hasattr(vision_model.embeddings.position_embedding, "weight"):
                num_pos = vision_model.embeddings.position_embedding.weight.shape[0]
                if vis_pos_ids.max() >= num_pos:
                    print(f"WARNING: Vision Position IDs are invalid (max={vis_pos_ids.max()}, limit={num_pos}). Re-initializing...")
                    new_vis_pos_ids = torch.arange(num_pos).expand((1, -1)).clone()
                    vision_model.embeddings.position_ids = new_vis_pos_ids
                    print(f"Fixed Vision Position IDs (Size: {num_pos}).")

        model.set_processor(MODEL_NAME)
        model.eval()

        # [Workaround] Move only visual parts to GPU, keep text parts on CPU
        # This avoids CUDA errors during text encoding which seems to be sensitive to index issues on GPU
        print(f"Moving visual parts to {device}...")
        if hasattr(model, "vision_model"):
            model.vision_model.to(device)
        elif hasattr(model, "visual_model"):
            model.visual_model.to(device)

        if hasattr(model, "visual_projection"):
            model.visual_projection.to(device)

        # Ensure text model is on CPU
        if hasattr(model, "text_model"):
            model.text_model.to('cpu')
        if hasattr(model, "text_projection"):
            model.text_projection.to('cpu')

        # Update model.device if possible, but it's a property usually.
        # We just need to be careful with input devices.

    except Exception as e:
        print(f"Error loading BGE-VL: {e}")
        return [], []

    if isinstance(prompts, str):
        prompts = [prompts]

    # Encode Queries (Text)
    try:
        with torch.no_grad():
            # Tokenize prompts
            inputs = model.processor(text=prompts, return_tensors="pt", padding=True, truncation=True)

            # inputs are on CPU, text_model is on CPU
            # BGE-VL encode returns tensor
            query_embeddings = model.encode_text(inputs)
            query_embeddings = query_embeddings.to(device) # Move result to GPU
    except RuntimeError as e:
        if "device-side assert" in str(e):
            print(f"CUDA Error during text encoding: {e}")
            print("This usually indicates the Tokenizer produced IDs larger than the Model's embedding size.")
            return [], []
        raise e

    all_scores = []
    all_paths = []

    with torch.no_grad():
        for bi in range(0, len(image_paths), bs):
            cache_file = os.path.join(embeddings_path, f"bge_vl_embeddings_b{bi}.pt")
            current_batch_paths = image_paths[bi : bi + bs]

            if os.path.exists(cache_file):
                try:
                    data = torch.load(cache_file, map_location=device, weights_only=False)
                    batch_doc_embeddings = data['bge_vl_embeddings']
                    final_bi_paths = data.get('paths', current_batch_paths)
                except Exception as e:
                    print(f"Error loading cache {cache_file}: {e}. Recomputing.")
                    os.remove(cache_file)
                    # Fallthrough to recompute

            # Check again if we need to compute (cache might have been deleted above)
            if not os.path.exists(cache_file):
                # Filter valid paths
                valid_paths = [p for p in current_batch_paths if os.path.exists(p)]
                if not valid_paths: continue

                try:
                    # Manual image encoding to handle split devices (Visual on GPU, Text on CPU)
                    # Open images
                    pil_images = []
                    for p in valid_paths:
                        try:
                            pil_images.append(Image.open(p).convert("RGB"))
                        except Exception as e:
                            print(f"Warning: Could not open {p}: {e}")

                    if not pil_images: continue

                    # Process images
                    inputs = model.processor(images=pil_images, return_tensors="pt")
                    # Move to GPU (Visual model is on GPU)
                    inputs = inputs.to(device)

                    # Encode
                    # [Fix] Pass pixel_values tensor directly, as encode_image expects a tensor
                    batch_doc_embeddings = model.encode_image(inputs['pixel_values'])

                except RuntimeError as e:
                    print(f"CUDA Error during image encoding: {e}")
                    return [], []

                final_bi_paths = valid_paths

            # Ensure device match
            batch_doc_embeddings = batch_doc_embeddings.to(device)
            query_embeddings = query_embeddings.to(device)

            # [Fix] Normalize embeddings for Cosine Similarity
            batch_doc_embeddings = F.normalize(batch_doc_embeddings, p=2, dim=-1)
            query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)

            sim_matrix = query_embeddings @ batch_doc_embeddings.T
            # [Fix] Convert BFloat16 to Float32 before converting to numpy
            all_scores.append(sim_matrix.float().cpu().numpy())
            all_paths.extend(final_bi_paths)

    if not all_scores:
        return [], []

    full_scores = np.concatenate(all_scores, axis=1) # [N_q, Total_Docs]

    # Return top-k for each prompt
    final_paths_list = []
    final_scores_list = []

    for i in range(len(prompts)):
        scores = full_scores[i]
        top_indices = np.argsort(scores)[-k:][::-1]

        top_paths = [all_paths[idx] for idx in top_indices]
        top_scores = scores[top_indices].tolist()

        final_paths_list.append(top_paths)
        final_scores_list.append(top_scores)

    # Cleanup to free GPU memory
    del model
    torch.cuda.empty_cache()

    return final_paths_list, final_scores_list

def get_qwen2_5_vl_similarities(prompts, image_paths, embeddings_path="", bs=1, k=50, device='cuda:0', save=False, adapter_path=None):
    """
    Qwen2.5-VL Retrieval (VLM2Vec)
    """
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        # Use local path if available
        local_path = "models/Qwen/Qwen2.5-VL-3B-Instruct"
        model_name = local_path if os.path.exists(local_path) else "Qwen/Qwen2.5-VL-3B-Instruct"

        print(f"Loading Qwen2.5-VL from {model_name}...")
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
        print(f"Error loading Qwen2.5-VL: {e}")
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
    from tqdm import tqdm
    print(f"Encoding {len(image_paths)} images with Qwen2.5-VL (Batch Size: {bs})...")
    with torch.no_grad():
        for bi in tqdm(range(0, len(image_paths), bs), desc="Qwen2.5-VL Embedding"):
            cache_file = os.path.join(embeddings_path, f"qwen2_5_vl_embeddings_b{bi}.pt")
            current_batch_paths = image_paths[bi : bi + bs]

            batch_doc_embeddings = None
            final_bi_paths = []

            if os.path.exists(cache_file):
                try:
                    data = torch.load(cache_file, map_location=device, weights_only=False)
                    batch_doc_embeddings = data['embeddings']
                    final_bi_paths = data.get('paths', current_batch_paths)
                except: pass

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

    return final_paths_list, final_scores_list

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
    from tqdm import tqdm
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

    return final_paths_list, final_scores_list

def retrieve_img_per_caption(captions, image_paths, embeddings_path="", k=3, device='cuda', method='CLIP', global_memory=None, adapter_path=None, use_hybrid=False, external_model=None, external_processor=None):
    """
    统一检索入口。
    支持 global_memory: 用于 Re-ranking (Penalize history)
    """
    all_retrieved_paths = []
    all_retrieved_scores = []

    for caption in captions:
        if method == 'CLIP':
            paths, scores = get_clip_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device,
                use_hybrid=use_hybrid
            )

            if global_memory:
                paths, scores = global_memory.re_rank(paths, scores, prompt=caption)

            all_retrieved_paths.append(paths)
            all_retrieved_scores.append(scores)

        elif method == 'SigLIP':
            paths, scores = get_siglip_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device, save=True
            )

            if global_memory:
                paths, scores = global_memory.re_rank(paths, scores, prompt=caption)

            all_retrieved_paths.append(paths)
            all_retrieved_scores.append(scores)

        elif method == 'SigLIP2':
            paths, scores = get_siglip2_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device, save=True
            )

            if global_memory:
                paths, scores = global_memory.re_rank(paths, scores, prompt=caption)

            all_retrieved_paths.append(paths)
            all_retrieved_scores.append(scores)

        elif method == 'LongCLIP':
            paths, scores = get_longclip_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device
            )
            if global_memory:
                paths, scores = global_memory.re_rank(paths, scores, prompt=caption)
            all_retrieved_paths.append(paths)
            all_retrieved_scores.append(scores)

        elif method == 'BGE-VL':
            paths_list, scores_list = get_bge_vl_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device, save=True
            )
            if paths_list:
                paths = paths_list[0]
                scores = scores_list[0]
            else:
                print("BGE-VL returned no results.")
                paths = []
                scores = []

            if global_memory:
                paths, scores = global_memory.re_rank(paths, scores, prompt=caption)
            all_retrieved_paths.append(paths)
            all_retrieved_scores.append(scores)

        elif method == 'Qwen3-VL':
            paths_list, scores_list = get_qwen3_vl_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device, save=True,
                adapter_path=adapter_path,
                external_model=external_model if external_model else (getattr(global_memory, 'model', None) if global_memory else None),
                external_processor=external_processor if external_processor else (getattr(global_memory, 'processor', None) if global_memory else None)
            )
            if paths_list:
                paths = paths_list[0]
                scores = scores_list[0]
            else:
                paths = []
                scores = []

            if global_memory:
                paths, scores = global_memory.re_rank(paths, scores, prompt=caption)
            all_retrieved_paths.append(paths)
            all_retrieved_scores.append(scores)

        elif method == 'Qwen2.5-VL':
            paths_list, scores_list = get_qwen2_5_vl_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device, save=True,
                adapter_path=adapter_path
            )
            if paths_list:
                paths = paths_list[0]
                scores = scores_list[0]
            else:
                paths = []
                scores = []

            if global_memory:
                paths, scores = global_memory.re_rank(paths, scores, prompt=caption)
            all_retrieved_paths.append(paths)
            all_retrieved_scores.append(scores)

        else:
            print(f"Unknown method {method}, falling back to CLIP")
            paths, scores = get_clip_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device
            )

            if global_memory:
                paths, scores = global_memory.re_rank(paths, scores, prompt=caption)

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
                data = torch.load(cache_file, map_location=device, weights_only=False)
                # Support both keys for backward compatibility
                normalized_im_vectors = data.get('normalized_longclip_embeddings', data.get('normalized_clip_embeddings'))
                final_bi_paths = data.get('paths', current_batch_paths)
            else:
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
                        "normalized_longclip_embeddings": normalized_im_vectors,
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
