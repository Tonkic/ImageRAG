import os
import torch
import clip
from open_clip import create_model_from_pretrained, get_tokenizer
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import AutoModel, AutoProcessor

# Global cache to prevent redundant model loading across multiple sub-retrievers (e.g., aircraft, cub, imagenet)
GLOBAL_RETRIEVAL_MODELS = {}
GLOBAL_RETRIEVAL_PREPROCESSORS = {}

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

# [Removed] Long-CLIP Imports / Pic2Word imports
# [Method retrieve_composed removed]

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'Long-CLIP')))
try:
    from model import longclip
    LONGCLIP_AVAILABLE = True
except ImportError:
    LONGCLIP_AVAILABLE = False
    print("Warning: Long-CLIP not found.")


def get_siglip2_model_id():
    return os.environ.get("SIGLIP2_MODEL_ID", "google/siglip2-base-patch16-224")


def _unwrap_siglip2_features(features, modality="image"):
    if isinstance(features, torch.Tensor):
        return features

    preferred = [f"{modality}_embeds", "pooler_output", "image_embeds", "text_embeds", "last_hidden_state"]
    for attr in preferred:
        value = getattr(features, attr, None)
        if isinstance(value, torch.Tensor):
            if value.dim() == 3:
                return value[:, 0, :]
            return value

    if isinstance(features, (tuple, list)) and features:
        first = features[0]
        if isinstance(first, torch.Tensor):
            if first.dim() == 3:
                return first[:, 0, :]
            return first

    raise TypeError(f"Unsupported SigLIP2 feature output type: {type(features)}")

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
    SigLIP2 Retrieval (configurable via SIGLIP2_MODEL_ID env var)
    """
    try:
        from transformers import AutoModel, AutoProcessor
        model_name = get_siglip2_model_id()
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
        text_features = _unwrap_siglip2_features(model.get_text_features(**text_inputs), modality="text")
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
                image_features = _unwrap_siglip2_features(model.get_image_features(**image_inputs), modality="image")
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


# -----------------------------------------------------------------------------
# [NEW] Optimized ImageRetriever Class
# -----------------------------------------------------------------------------
class ImageRetriever:
    def __init__(self, image_paths, embeddings_path, method="CLIP", device="cuda", k=10, use_hybrid=False,
                 global_memory=None, external_model=None, external_processor=None, adapter_path=None):
        self.method = method
        self.device = device
        self.k = k
        self.use_hybrid = use_hybrid
        self.global_memory = global_memory
        self.image_paths = image_paths
        self.embeddings_path = embeddings_path
        self.siglip2_model_name = None
        self.embedding_model_name = None

        # Model & Processor
        self.model = None
        self.processor = None
        self.preprocess = None
        self.tokenizer = None

        # Data
        self.image_embeddings = None # Tensor [N, Dim]
        self.valid_paths = []        # List of paths matching rows in image_embeddings

        # Performance: Preload everything
        self._load_model(external_model, external_processor, adapter_path)
        self._load_database()
        self._align_siglip2_model_to_embeddings()

        # Hybrid Setup
        self.bm25_retriever = None
        if self.use_hybrid:
            print(f"[ImageRetriever] Initializing HybridRetriever (BM25)...")
            self.bm25_retriever = HybridRetriever(self.valid_paths)

    def _load_model(self, ext_model, ext_proc, adapter_path):
        global GLOBAL_RETRIEVAL_MODELS, GLOBAL_RETRIEVAL_PREPROCESSORS

        # Build unique cache key
        if self.method == "SigLIP2":
            requested_model_name = get_siglip2_model_id()
            cache_key = f"{self.method}_{requested_model_name}_{self.device}"
        else:
            cache_key = f"{self.method}_{self.device}"

        # Core logic: If already loaded, borrow from memory!
        if cache_key in GLOBAL_RETRIEVAL_MODELS:
            print(f"[ImageRetriever] ♻️ Using cached {self.method} model on {self.device} to save VRAM!")
            self.model = GLOBAL_RETRIEVAL_MODELS[cache_key]
            self.preprocess = GLOBAL_RETRIEVAL_PREPROCESSORS[cache_key]

            # For SigLIP and others that have tokenizer/processor, pull them out if they are stored in preprocess
            # (or we can just reuse self.model/self.preprocess as they are)
            if self.method == "SigLIP":
                self.tokenizer = self.preprocess
            elif self.method == "SigLIP2":
                self.processor = self.preprocess
            elif self.method in ["Qwen3-VL", "Qwen2.5-VL"]:
                self.processor = self.preprocess

            return

        print(f"[ImageRetriever] Loading Model: {self.method}...")

        if self.method == "CLIP":
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        elif self.method == "LongCLIP":
            if not LONGCLIP_AVAILABLE:
                raise ImportError(
                    "LongCLIP module is not available. Please ensure Long-CLIP is installed under "
                    f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'Long-CLIP'))}."
                )
            ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'Long-CLIP', 'checkpoints', 'longclip-L.pt'))
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"LongCLIP checkpoint missing at {ckpt_path}. "
                    f"Please download it before using --retrieval_method LongCLIP."
                )
            self.model, self.preprocess = longclip.load(ckpt_path, device=self.device)
            self.model.eval()

        elif self.method == "SigLIP":
            self.model, self.preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384', device=self.device)
            self.tokenizer = get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
            self.preprocess = self.tokenizer   # Store tokenizer as preprocess for caching sake

        elif self.method == "SigLIP2":
            self._load_siglip2_model_by_name(requested_model_name)
            return

        elif self.method in ["Qwen3-VL", "Qwen2.5-VL"]:
            # Use external if provided to save VRAM
            if ext_model is not None:
                print("  Using shared external model.")
                self.model = ext_model
                self.processor = ext_proc
                self.preprocess = ext_proc
            else:
                 # Load logic mainly for fallback, usually passed externally in script
                 print("  Warning: No external model passed for VLM retrieval. Loading new instance.")
                 pass

        # Save to global cache
        GLOBAL_RETRIEVAL_MODELS[cache_key] = self.model
        GLOBAL_RETRIEVAL_PREPROCESSORS[cache_key] = self.preprocess

    def _load_siglip2_model_by_name(self, model_name):
        global GLOBAL_RETRIEVAL_MODELS, GLOBAL_RETRIEVAL_PREPROCESSORS

        cache_key = f"SigLIP2_{model_name}_{self.device}"
        if cache_key in GLOBAL_RETRIEVAL_MODELS:
            print(f"[ImageRetriever] ♻️ Using cached SigLIP2 model {model_name} on {self.device}.")
            self.model = GLOBAL_RETRIEVAL_MODELS[cache_key]
            self.preprocess = GLOBAL_RETRIEVAL_PREPROCESSORS[cache_key]
            self.processor = self.preprocess
            self.siglip2_model_name = model_name
            return

        from transformers import AutoModel, AutoProcessor

        print(f"[ImageRetriever] Loading SigLIP2 model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name, attn_implementation="sdpa").to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.preprocess = self.processor
        self.siglip2_model_name = model_name

        GLOBAL_RETRIEVAL_MODELS[cache_key] = self.model
        GLOBAL_RETRIEVAL_PREPROCESSORS[cache_key] = self.preprocess

    def _align_siglip2_model_to_embeddings(self):
        if self.method != "SigLIP2":
            return

        if not self.embedding_model_name:
            return

        if self.siglip2_model_name == self.embedding_model_name:
            return

        print(
            "[ImageRetriever] SigLIP2 model/cache mismatch detected: "
            f"query_model={self.siglip2_model_name}, cache_model={self.embedding_model_name}. "
            "Switching query encoder to match cache model."
        )
        self._load_siglip2_model_by_name(self.embedding_model_name)

    def _load_database(self):
        print(f"[ImageRetriever] Loading Database Embeddings from {self.embeddings_path}...")
        all_tensors = []
        all_paths = []
        expected_dim = None
        siglip2_model_names = set()

        # Iterate all potential batch files
        # Heuristic: Check up to 1000 batches or until not found?
        # Better: iterate image_paths with same batch logic to align files.
        bs = 1024 # Standard cache bs

        # Identify cache prefix
        prefix_map = {
            "CLIP": "clip_embeddings",
            "LongCLIP": "longclip_embeddings",
            "SigLIP": "siglip_embeddings",
            "SigLIP2": "siglip2_embeddings",
            "Qwen3-VL": "qwen3_vl_embeddings",
            "Qwen2.5-VL": "qwen2_5_vl_embeddings"
        }
        prefix = prefix_map.get(self.method, "clip_embeddings")

        # We process in the same chunks as creation to find files
        for bi in range(0, len(self.image_paths), bs):
            cache_file = os.path.join(self.embeddings_path, f"{prefix}_b{bi}.pt")
            if os.path.exists(cache_file):
                try:
                    data = torch.load(cache_file, map_location="cpu", weights_only=False) # Load to CPU first

                    # Extract Tensor
                    # Handle various key names from legacy code
                    keys = [f"normalized_{prefix}", "embeddings", f"{prefix}"]
                    # Add fallback for specific legacy keys
                    if self.method == "CLIP": keys.append("normalized_clip_embeddings")

                    emb = None
                    for k in keys:
                        if k in data:
                            emb = data[k]
                            break

                    if emb is None:
                         # Fallback search
                         for v in data.values():
                             if isinstance(v, torch.Tensor):
                                 emb = v
                                 break

                    paths = data.get("paths", [])

                    if self.method == "SigLIP2":
                        model_name = data.get("model_name") or data.get("model_id")
                        if isinstance(model_name, str) and model_name.strip():
                            siglip2_model_names.add(model_name.strip())

                    if emb is not None and len(paths) > 0:
                        if emb.dim() != 2:
                            print(f"[ImageRetriever] Skip malformed embedding tensor in {cache_file}, shape={tuple(emb.shape)}")
                            continue

                        if expected_dim is None:
                            expected_dim = emb.shape[1]
                        elif emb.shape[1] != expected_dim:
                            print(
                                f"[ImageRetriever] Skip {cache_file} due to dim mismatch: "
                                f"{emb.shape[1]} vs expected {expected_dim}"
                            )
                            continue

                        if emb.shape[0] != len(paths):
                            n = min(emb.shape[0], len(paths))
                            print(
                                f"[ImageRetriever] Path/embedding count mismatch in {cache_file}: "
                                f"emb={emb.shape[0]} paths={len(paths)}. Truncating to {n}."
                            )
                            emb = emb[:n]
                            paths = paths[:n]

                        all_tensors.append(emb) # Keep on CPU for concatenation
                        all_paths.extend(paths)
                except Exception as e:
                    print(f"Error loading {cache_file}: {e}")

        if self.method == "SigLIP2" and siglip2_model_names:
            if len(siglip2_model_names) > 1:
                print(
                    "[ImageRetriever] WARNING: mixed SigLIP2 cache model names detected: "
                    f"{sorted(siglip2_model_names)}. Using first one for query alignment."
                )
            self.embedding_model_name = sorted(siglip2_model_names)[0]

        if not all_tensors:
            print("WARNING: No embeddings loaded! Retrieval will fail.")
            self.image_embeddings = None
            return

        # Concatenate
        try:
            full_tensor = torch.cat(all_tensors, dim=0)
            self.image_embeddings = full_tensor.to(self.device) # Move to GPU once
            self.valid_paths = all_paths
            print(f"[ImageRetriever] Loaded {len(self.valid_paths)} images into memory. Tensor shape: {self.image_embeddings.shape}")
        except Exception as e:
             print(f"Error concatenating embeddings: {e}")

    def search(self, queries):
        """
        Fast batch search.
        """
        if self.image_embeddings is None:
            return [], []

        if isinstance(queries, str): queries = [queries]

        # 1. Encode Queries
        query_embs = self._encode_text(queries) # [B, Dim]

        if query_embs is None:
            return [], []

        if query_embs.shape[-1] != self.image_embeddings.shape[-1]:
            raise RuntimeError(
                "Retriever embedding dim mismatch: "
                f"query_dim={query_embs.shape[-1]} vs db_dim={self.image_embeddings.shape[-1]}. "
                f"method={self.method}, siglip2_query_model={self.siglip2_model_name}, "
                f"siglip2_cache_model={self.embedding_model_name}."
            )

        if query_embs.dtype != self.image_embeddings.dtype:
            if str(self.device).startswith("cpu"):
                if self.image_embeddings.dtype != torch.float32:
                    self.image_embeddings = self.image_embeddings.float()
                query_embs = query_embs.float()
            else:
                query_embs = query_embs.to(self.image_embeddings.dtype)

        # 2. Dot Product
        # query_embs: [B, D], image_embeddings: [N, D] -> [B, N]
        sim_matrix = torch.matmul(query_embs, self.image_embeddings.T)

        # 3. Hybrid
        if self.use_hybrid and self.bm25_retriever:
            # Note: Hybrid is slow if batched naively.
            # Process row-by-row
            hybrid_scores_list = []
            sim_np = sim_matrix.cpu().numpy()

            for i, q in enumerate(queries):
                vec_s = sim_np[i]
                final_s = self.bm25_retriever.hybrid_search(q, vec_s)
                hybrid_scores_list.append(final_s)

            sim_matrix = torch.tensor(np.stack(hybrid_scores_list), device=self.device)

        # 4. Top-K
        # If we have Global Memory, we might need more candidates initially?
        # For simplicity, get Top-K * 2 then rerank

        topk_val, topk_idx = torch.topk(sim_matrix, k=min(self.k * 2, len(self.valid_paths)), dim=1)

        topk_val = topk_val.float().cpu().numpy()
        topk_idx = topk_idx.cpu().numpy()

        results_paths = []
        results_scores = []

        for i in range(len(queries)):
            indices = topk_idx[i]
            scores = topk_val[i]

            paths = [self.valid_paths[idx] for idx in indices]
            score_list = scores.tolist()

            # Global Memory Rerank
            if self.global_memory:
                 paths, score_list = self.global_memory.re_rank(paths, score_list, prompt=queries[i])
                 # Truncate to k after rerank
                 paths = paths[:self.k]
                 score_list = score_list[:self.k]

            results_paths.append(paths)
            results_scores.append(score_list)

        return results_paths, results_scores

    def _encode_text(self, queries):
        with torch.no_grad():
            if self.method == "CLIP":
                import clip
                tokens = clip.tokenize(queries, truncate=True).to(self.device)
                feats = self.model.encode_text(tokens)
                return F.normalize(feats, p=2, dim=-1)

            elif self.method == "LongCLIP":
                from model import longclip
                tokens = longclip.tokenize(queries, truncate=True).to(self.device)
                feats = self.model.encode_text(tokens)
                return F.normalize(feats, p=2, dim=-1)

            elif self.method == "SigLIP":
                # uses tokenizer
                inputs = self.tokenizer(queries, context_length=self.model.context_length).to(self.device)
                feats = self.model.encode_text(inputs)
                return F.normalize(feats, p=2, dim=-1)

            elif self.method == "SigLIP2":
                text_inputs = self.processor(text=queries, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
                feats = _unwrap_siglip2_features(self.model.get_text_features(**text_inputs), modality="text")
                return F.normalize(feats, dim=-1)

            elif self.method in ["Qwen3-VL", "Qwen2.5-VL"]:
                # Chat template encoding
                embs = []
                for q in queries:
                    msgs = [{"role": "user", "content": [{"type": "text", "text": f"Retrieve the image that matches the description: {q}"}]}]
                    inputs = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.device)
                    out = self.model(**inputs, output_hidden_states=True)
                    emb = out.hidden_states[-1][:, -1, :]
                    embs.append(emb)
                return F.normalize(torch.cat(embs, dim=0), p=2, dim=-1)

        return None

# Wrapper for Backward Compatibility
GLOBAL_RETRIEVER_INSTANCE = None

def retrieve_img_per_caption(captions, image_paths, embeddings_path="", k=3, device='cuda', method='CLIP', global_memory=None, adapter_path=None, use_hybrid=False, external_model=None, external_processor=None):
    """
    Optimized Wrapper: Uses a persistent GLOBAL_RETRIEVER_INSTANCE if available, or creates one.
    To benefit from speedup, the main script should ideally instantiate ImageRetriever directly.
    But this singleton approach helps legacy calls too.
    """
    global GLOBAL_RETRIEVER_INSTANCE

    # Check if we need to re-init (e.g. method changed, or not created)
    # Simple check: if None or method mismatch
    if GLOBAL_RETRIEVER_INSTANCE is None or GLOBAL_RETRIEVER_INSTANCE.method != method:
        print(f"[Wrapper] Initializing Global ImageRetriever for {method}...")
        GLOBAL_RETRIEVER_INSTANCE = ImageRetriever(
            image_paths=image_paths,
            embeddings_path=embeddings_path,
            method=method,
            device=device,
            k=k,
            use_hybrid=use_hybrid,
            global_memory=global_memory,
            external_model=external_model,
            external_processor=external_processor,
            adapter_path=adapter_path
        )

    # Update runtime params if needed (e.g. k might change per call)
    GLOBAL_RETRIEVER_INSTANCE.k = k
    GLOBAL_RETRIEVER_INSTANCE.global_memory = global_memory

    return GLOBAL_RETRIEVER_INSTANCE.search(captions)

