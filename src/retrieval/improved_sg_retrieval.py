#!/usr/bin/env python3
"""
Improved Semantic Gap Retrieval with Two-Stage Selection
Stage 1: Use text similarity to filter semantically correct candidates
Stage 2: Select highest gap from filtered candidates for visual diversity
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import clip

class ImprovedSemanticGapRetriever:
    """
    Two-stage retrieval:
    1. Filter by text similarity (ensure semantic correctness)
    2. Select by gap (maximize visual-semantic diversity)
    """
    def __init__(self, embedding_matrix, image_paths, label_file, clip_model, device='cuda'):
        self.device = device
        self.clip_model = clip_model
        self.visual_means = {}  # Class -> Tensor [D]
        self.text_protos = {}   # Class -> Tensor [D]

        print(f"[ImprovedSG] Fitting Visual and Text Prototypes on {len(image_paths)} images...")

        # 1. Parse Labels
        valid_classes = set()
        id_to_class = {}
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        id_to_class[parts[0]] = parts[1]
                        valid_classes.add(parts[1])
        else:
            print(f"Error: Label file {label_file} not found. ImprovedSG cannot work.")
            return

        # 2. Visual Means
        class_embeddings = {}
        if isinstance(embedding_matrix, torch.Tensor):
            embeddings = embedding_matrix.cpu().numpy()
        else:
            embeddings = embedding_matrix

        for i, path in enumerate(image_paths):
            img_id = os.path.splitext(os.path.basename(path))[0]
            if img_id in id_to_class:
                cls = id_to_class[img_id]
                if cls not in class_embeddings:
                    class_embeddings[cls] = []
                class_embeddings[cls].append(embeddings[i])

        for cls, vecs in class_embeddings.items():
            vecs = np.array(vecs)
            mu = np.mean(vecs, axis=0)  # [D]
            mu = mu / np.linalg.norm(mu)  # Normalize prototype
            self.visual_means[cls] = torch.tensor(mu, device=device).float()

        # 3. Text Prototypes
        print(f"[ImprovedSG] Computing Text Prototypes for {len(valid_classes)} classes...")
        with torch.no_grad():
            for cls in valid_classes:
                text = f"a photo of a {cls}"
                tokenized = clip.tokenize([text]).to(device)
                # [LongCLIP Fix] Pad to 248 if needed
                if tokenized.shape[-1] < 248:
                    pad_len = 248 - tokenized.shape[-1]
                    tokenized = torch.cat([tokenized, torch.zeros((tokenized.shape[0], pad_len), dtype=tokenized.dtype, device=device)], dim=1)

                text_feat = clip_model.encode_text(tokenized)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                self.text_protos[cls] = text_feat.squeeze(0).float()

    def retrieve_two_stage(self, feats_tensor, target_class, stage1_k=10, verbose=False):
        """
        Two-stage retrieval:
        Stage 1: Select top-K by text similarity (semantic correctness)
        Stage 2: Select highest gap from stage1 candidates (visual diversity)

        Args:
            feats_tensor: [N, D] normalized features of all candidates
            target_class: str, target class name
            stage1_k: int, number of candidates to keep after stage 1
            verbose: bool, print debug info

        Returns:
            best_idx: int, index of selected candidate
            gap_score: float, gap score of selected candidate
        """
        if target_class not in self.visual_means or target_class not in self.text_protos:
            if verbose:
                print(f"[ImprovedSG] Unknown class: {target_class}")
            return None, None

        # Ensure Float32
        feats_tensor = feats_tensor.float()

        vis_proto = self.visual_means[target_class]
        text_proto = self.text_protos[target_class]

        # Compute similarities
        sim_v = feats_tensor @ vis_proto  # [N]
        sim_t = feats_tensor @ text_proto  # [N]

        # Stage 1: Filter by text similarity
        # Select top-K candidates with highest text similarity
        topk_values, topk_indices = torch.topk(sim_t, k=min(stage1_k, len(sim_t)))

        if verbose:
            print(f"[ImprovedSG] Stage 1: Selected top-{len(topk_indices)} by text similarity")
            print(f"   Text sim range: [{topk_values[-1]:.4f}, {topk_values[0]:.4f}]")

        # Stage 2: Select highest gap from stage1 candidates
        # Gap = |sim_v - sim_t|
        stage1_sim_v = sim_v[topk_indices]
        stage1_sim_t = sim_t[topk_indices]
        stage1_gaps = torch.abs(stage1_sim_v - stage1_sim_t)

        # Find candidate with highest gap
        best_stage2_idx = torch.argmax(stage1_gaps)
        best_original_idx = topk_indices[best_stage2_idx].item()
        best_gap = stage1_gaps[best_stage2_idx].item()

        if verbose:
            print(f"[ImprovedSG] Stage 2: Selected highest gap from stage1 candidates")
            print(f"   Gap range: [{stage1_gaps.min():.4f}, {stage1_gaps.max():.4f}]")
            print(f"   Selected: idx={best_original_idx}, gap={best_gap:.4f}")
            print(f"   Selected sim_v={sim_v[best_original_idx]:.4f}, sim_t={sim_t[best_original_idx]:.4f}")

        return best_original_idx, best_gap
