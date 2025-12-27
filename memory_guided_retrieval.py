import os
import torch
import clip
from open_clip import create_model_from_pretrained, get_tokenizer
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import AutoModel, AutoProcessor

# [Optional] Long-CLIP Imports
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Long-CLIP'))
try:
    from model import longclip
    LONGCLIP_AVAILABLE = True
except ImportError:
    LONGCLIP_AVAILABLE = False
    print("Warning: Long-CLIP not found.")

# [Optional] ColPali Imports
try:
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    from transformers.utils.import_utils import is_flash_attn_2_available
    COLPALI_AVAILABLE = True
except ImportError:
    COLPALI_AVAILABLE = False
    print("Warning: ColPali engine not found. method='ColPali' will fail.")

# -----------------------------------------------------------------------------
# Memory-Guided Retrieval Module
# -----------------------------------------------------------------------------

def check_token_length(prompts, device="cpu", method="CLIP"):
    # [Modified] Disable warning as we now handle long text via chunking
    return

def get_colpali_similarities(prompts, image_paths, embeddings_path="", bs=4, k=50, device='cuda:0', save=False):
    """
    ColPali (ColQwen2.5) Retrieval
    """
    if not COLPALI_AVAILABLE:
        print("ColPali not available.")
        return [], []

    model_name = "vidore/colqwen2.5-v0.2"
    try:
        processor = ColQwen2_5_Processor.from_pretrained(model_name)
        model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
    except Exception as e:
        print(f"Error loading ColPali: {e}")
        return [], []

    if isinstance(prompts, str):
        prompts = [prompts]

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

                # Process Images
                batch_images = processor.process_images(images).to(device)
                # [Batch, N_d, Dim]
                batch_doc_embeddings = model(**batch_images)

                # Save as list to handle variable sequence lengths if any (though ColQwen usually fixed patches)
                # But let's save the tensor batch if possible, or list
                # ColQwen output is a list of tensors if input is list? No, model(**batch) returns tensor usually?
                # Actually ColQwen2.5 model forward returns embeddings.

                final_bi_paths = valid_paths

                if embeddings_path:
                    os.makedirs(embeddings_path, exist_ok=True)
                    torch.save({
                        "colpali_embeddings": batch_doc_embeddings,
                        "paths": final_bi_paths
                    }, cache_file)
            else:
                continue

            # Compute MaxSim (Late Interaction)
            # Q: [1, N_q, D], D: [B, N_d, D]
            # Score = sum_over_q(max_over_d(Q @ D.T))

            # Ensure dtype match
            batch_doc_embeddings = batch_doc_embeddings.to(query_embeddings.dtype)

            # Loop over queries
            for i in range(len(prompts)):
                Q = query_embeddings[i] # [N_q, D]
                # Batch of docs: [B, N_d, D]
                D = batch_doc_embeddings

                # Q: [N_q, D] -> [1, N_q, D]
                # D: [B, N_d, D] -> [B, D, N_d] (transpose last two)

                # Sim: [B, N_q, N_d]
                sim_matrix = torch.einsum('qd,bnd->bqn', Q, D)

                # Max over doc tokens: [B, N_q]
                max_sim_scores = sim_matrix.max(dim=-1).values

                # Sum over query tokens: [B]
                scores = max_sim_scores.sum(dim=-1)

                all_scores.append(scores.cpu().numpy()) # This appends [B] array

            all_paths.extend(final_bi_paths)

    # Re-organize scores
    # all_scores is a list of arrays, one per batch? No, I looped over queries inside batch loop.
    # Wait, the structure above is wrong.
    # I should accumulate scores for each query across all batches.

    # Let's restructure:
    # Initialize scores for each query
    final_scores_map = {i: [] for i in range(len(prompts))}
    final_paths = [] # Assuming all batches processed sequentially

    # ... (Re-implementing loop logic correctly in the tool call)

def get_clip_similarities(prompts, image_paths, embeddings_path="", bs=1024, k=50, device='cuda:0'):
    """
    CLIP 检索核心逻辑 (支持长文本 Chunking)
    """
    model, preprocess = clip.load("ViT-B/32", device=device)

    if isinstance(prompts, str):
        prompts = [prompts]

    # [Modified] Long Text Handling: Chunk & Average
    text_features_list = []
    with torch.no_grad():
        for p in prompts:
            try:
                # Try full text first (truncate=False to detect overflow)
                text = clip.tokenize([p], truncate=False).to(device)
                feat = model.encode_text(text)
            except RuntimeError:
                # If too long, chunk by words
                words = p.split()
                chunks = []
                chunk_size = 50 # Safe margin for 77 token limit
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i+chunk_size])
                    chunks.append(chunk)

                if not chunks: chunks = [""]

                # Encode chunks and average
                text_chunks = clip.tokenize(chunks, truncate=True).to(device)
                chunk_feats = model.encode_text(text_chunks) # [N, 512]
                feat = chunk_feats.mean(dim=0, keepdim=True) # [1, 512]

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

    k = min(k, len(single_prompt_scores))
    top_indices = np.argsort(single_prompt_scores)[-k:][::-1]

    top_paths = [all_paths[i] for i in top_indices]
    top_scores = single_prompt_scores[top_indices].tolist()

    return top_paths, top_scores


    # Re-organize scores
    # We need to handle the batching correctly.
    # Let's simplify: return empty for now if not implemented fully, but I will implement fully below.
    pass

def get_colpali_similarities_full(prompts, image_paths, embeddings_path="", bs=4, k=50, device='cuda:0', save=False):
    if not COLPALI_AVAILABLE: return [], []

    model_name = "vidore/colqwen2.5-v0.2"
    try:
        processor = ColQwen2_5_Processor.from_pretrained(model_name)
        model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
    except: return [], []

    if isinstance(prompts, str): prompts = [prompts]

    with torch.no_grad():
        batch_queries = processor.process_queries(prompts).to(device)
        query_embeddings = model(**batch_queries) # [N_q_batch, Seq_q, Dim]

    all_scores_per_query = [[] for _ in range(len(prompts))]
    all_paths = []

    with torch.no_grad():
        for bi in range(0, len(image_paths), bs):
            cache_file = os.path.join(embeddings_path, f"colpali_embeddings_b{bi}.pt")
            current_batch_paths = image_paths[bi : bi + bs]

            batch_doc_embeddings = None
            final_bi_paths = []

            if os.path.exists(cache_file):
                try:
                    data = torch.load(cache_file, map_location=device, weights_only=False)
                    batch_doc_embeddings = data['colpali_embeddings']
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

                if images:
                    batch_images = processor.process_images(images).to(device)
                    batch_doc_embeddings = model(**batch_images) # [B, Seq_d, Dim]
                    final_bi_paths = valid_paths

                    if embeddings_path:
                        os.makedirs(embeddings_path, exist_ok=True)
                        torch.save({"colpali_embeddings": batch_doc_embeddings, "paths": final_bi_paths}, cache_file)

            if batch_doc_embeddings is not None:
                batch_doc_embeddings = batch_doc_embeddings.to(query_embeddings.dtype)
                # MaxSim
                # Q: [N_prompts, Seq_q, Dim]
                # D: [B, Seq_d, Dim]
                for i in range(len(prompts)):
                    Q = query_embeddings[i] # [Seq_q, Dim]
                    D = batch_doc_embeddings # [B, Seq_d, Dim]

                    # [B, Seq_q, Seq_d]
                    sim = torch.einsum('qd,bnd->bqn', Q, D)
                    scores = sim.max(dim=-1).values.sum(dim=-1) # [B]
                    all_scores_per_query[i].extend(scores.cpu().float().numpy().tolist())

                all_paths.extend(final_bi_paths)

    # Top K
    if not all_scores_per_query[0]: return [], []

    single_prompt_scores = np.array(all_scores_per_query[0])
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


def get_colqwen3_similarities(prompts, image_paths, embeddings_path="", bs=4, k=50, device='cuda:0', save=False):
    """
    ColQwen3 Retrieval (TomoroAI/tomoro-colqwen3-embed-8b)
    """
    print("Loading ColQwen3...")
    # Set HF Mirror
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    try:
        from transformers import AutoModel, AutoProcessor, AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        model_name = "TomoroAI/tomoro-colqwen3-embed-8b"

        # [Fix] Patch tie_weights
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            class_ref = config.auto_map["AutoModel"]
            model_class = get_class_from_dynamic_module(class_ref, model_name)
            if hasattr(model_class, "tie_weights"):
                original_tie_weights = model_class.tie_weights
                def safe_tie_weights(self, **kwargs):
                    kwargs.pop("missing_keys", None)
                    return original_tie_weights(self) # Removed **kwargs as per check script
                model_class.tie_weights = safe_tie_weights

            print(f"Loading ColQwen3 with BF16 and FlashAttention2 on {device}...")
            model = model_class.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": device}
            ).eval()
        except Exception as patch_err:
            print(f"Patching/Loading failed ({patch_err}), trying standard load...")
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, dtype="auto").to(device).eval()

        # [Refinement] Set max_num_visual_tokens=1280 as per official usage
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, max_num_visual_tokens=1280)
    except Exception as e:
        print(f"Error loading ColQwen3: {e}")
        return [], []

    if isinstance(prompts, str):
        prompts = [prompts]

    # Process Queries
    with torch.no_grad():
        # ColQwen3 uses process_texts
        if hasattr(processor, "process_texts"):
            batch_queries = processor.process_texts(prompts)
            # Move to device
            batch_queries = {k: v.to(device) for k, v in batch_queries.items()}
            query_outputs = model(**batch_queries)
            query_embeddings = query_outputs.embeddings
        elif hasattr(processor, "process_queries"): # ColPali fallback
            batch_queries = processor.process_queries(prompts).to(device)
            query_embeddings = model(**batch_queries)
        else:
            # Fallback or standard usage if processor handles it
            inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
            query_embeddings = model(**inputs)
            if hasattr(query_embeddings, 'embeddings'): # ColPali output
                query_embeddings = query_embeddings.embeddings
            elif hasattr(query_embeddings, 'last_hidden_state'):
                 query_embeddings = query_embeddings.last_hidden_state

    results_per_prompt = {i: {'paths': [], 'scores': []} for i in range(len(prompts))}

    with torch.no_grad():
        for bi in range(0, len(image_paths), bs):
            cache_file = os.path.join(embeddings_path, f"colqwen3_embeddings_b{bi}.pt")
            current_batch_paths = image_paths[bi : bi + bs]

            batch_doc_embeddings = None
            final_bi_paths = []

            if os.path.exists(cache_file):
                try:
                    data = torch.load(cache_file, map_location=device, weights_only=False)
                    loaded_embeddings = data['embeddings']
                    # [Fix] Dimension Check
                    if loaded_embeddings.shape[-1] != query_embeddings.shape[-1]:
                        print(f"[Batch {bi}/{len(image_paths)}] Dimension mismatch: Cache {loaded_embeddings.shape[-1]} vs Model {query_embeddings.shape[-1]}. Regenerating...")
                        batch_doc_embeddings = None # Force regen
                    else:
                        batch_doc_embeddings = loaded_embeddings
                        final_bi_paths = data.get('paths', current_batch_paths)
                except Exception as e:
                    print(f"Error loading cache {cache_file}: {e}. Regenerating...")
                    batch_doc_embeddings = None

            if batch_doc_embeddings is None and save:
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
                if hasattr(processor, "process_images"):
                    batch_images = processor.process_images(images)
                    # [Refinement] Robust device move
                    batch_images = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_images.items()}
                    image_outputs = model(**batch_images)
                    batch_doc_embeddings = image_outputs.embeddings
                else:
                    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
                    batch_doc_embeddings = model(**inputs)
                    if hasattr(batch_doc_embeddings, 'embeddings'):
                        batch_doc_embeddings = batch_doc_embeddings.embeddings
                    elif hasattr(batch_doc_embeddings, 'last_hidden_state'):
                        batch_doc_embeddings = batch_doc_embeddings.last_hidden_state

                final_bi_paths = valid_paths

                if embeddings_path:
                    os.makedirs(embeddings_path, exist_ok=True)
                    torch.save({
                        "embeddings": batch_doc_embeddings,
                        "paths": final_bi_paths
                    }, cache_file)

            if batch_doc_embeddings is None:
                continue

            # Compute Similarity (Late Interaction)
            batch_doc_embeddings = batch_doc_embeddings.to(query_embeddings.dtype)

            for i in range(len(prompts)):
                Q = query_embeddings[i] # [L_q, D]
                D = batch_doc_embeddings # [B, L_d, D]

                # Sim: [B, L_q, L_d]
                sim_matrix = torch.einsum('qd,bnd->bqn', Q, D)
                # Max over doc tokens: [B, L_q]
                max_sim_scores = sim_matrix.max(dim=-1).values
                # Sum over query tokens: [B]
                scores = max_sim_scores.sum(dim=-1)

                results_per_prompt[i]['paths'].extend(final_bi_paths)
                results_per_prompt[i]['scores'].extend(scores.cpu().tolist())

    # Sort and truncate
    final_paths_list = []
    final_scores_list = []

    for i in range(len(prompts)):
        paths = results_per_prompt[i]['paths']
        scores = results_per_prompt[i]['scores']

        combined = sorted(zip(paths, scores), key=lambda x: x[1], reverse=True)
        combined = combined[:k]

        p, s = zip(*combined) if combined else ([], [])
        final_paths_list.append(list(p))
        final_scores_list.append(list(s))

    return final_paths_list, final_scores_list

def retrieve_img_per_caption(captions, image_paths, embeddings_path="", k=3, device='cuda', method='CLIP', global_memory=None):
    """
    统一检索入口。
    支持 method: 'CLIP', 'SigLIP', 'Hybrid' (RRF Fusion)
    支持 global_memory: 用于 Re-ranking (Penalize history)
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

            # [Global Memory Re-ranking]
            # Apply before truncation to k, to allow lower-ranked items to bubble up if top ones are penalized
            if global_memory:
                # Convert RRF scores to list for re-ranking
                temp_scores = [rrf_scores[p] * 30.0 for p in sorted_paths]
                # Only use re_rank to adjust scores based on memory model, do NOT exclude
                sorted_paths, temp_scores = global_memory.re_rank(sorted_paths, temp_scores)
                # Update final scores map for the truncated list
                # (Actually we just need the sorted list and scores)
                final_scores = temp_scores[:k]
                sorted_paths = sorted_paths[:k]
            else:
                sorted_paths = sorted_paths[:k]
                final_scores = [rrf_scores[p] * 30.0 for p in sorted_paths]

            all_retrieved_paths.append(sorted_paths)
            all_retrieved_scores.append(final_scores)

        elif method == 'CLIP':
            paths, scores = get_clip_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device
            )

            if global_memory:
                paths, scores = global_memory.re_rank(paths, scores)

            all_retrieved_paths.append(paths)
            all_retrieved_scores.append(scores)

        elif method == 'SigLIP':
            paths, scores = get_siglip_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device, save=True
            )

            if global_memory:
                paths, scores = global_memory.re_rank(paths, scores)

            all_retrieved_paths.append(paths)
            all_retrieved_scores.append(scores)

        elif method == 'SigLIP2':
            paths, scores = get_siglip2_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device, save=True
            )

            if global_memory:
                paths, scores = global_memory.re_rank(paths, scores)

            all_retrieved_paths.append(paths)
            all_retrieved_scores.append(scores)

        elif method == 'ColPali':
            paths, scores = get_colpali_similarities_full(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device, save=True
            )
            if global_memory:
                paths, scores = global_memory.re_rank(paths, scores)
            all_retrieved_paths.append(paths)
            all_retrieved_scores.append(scores)

        elif method == 'LongCLIP':
            paths, scores = get_longclip_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device
            )
            if global_memory:
                paths, scores = global_memory.re_rank(paths, scores)
            all_retrieved_paths.append(paths)
            all_retrieved_scores.append(scores)

        elif method == 'colqwen3':
            paths_list, scores_list = get_colqwen3_similarities(
                [caption], image_paths,
                embeddings_path=embeddings_path,
                k=k, device=device, save=True
            )
            paths = paths_list[0]
            scores = scores_list[0]

            if global_memory:
                paths, scores = global_memory.re_rank(paths, scores)
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
                paths, scores = global_memory.re_rank(paths, scores)

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
                normalized_im_vectors = data['normalized_clip_embeddings']
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
