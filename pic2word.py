import torch
import torch.nn as nn
from collections import OrderedDict

# 1. 映射网络 (完全复刻 Pic2Word model.py)
class IM2TEXT(nn.Module):
    def __init__(self, embed_dim=1024, middle_dim=512, output_dim=1024, n_layer=2, dropout=0.1):
        """
        embed_dim: CLIP Image Embedding 维度 (ViT-L/14 为 768, ViT-H/14 为 1024)
        output_dim: CLIP Text Token Embedding 维度 (通常与 embed_dim 相同)
        """
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)

# 2. Token 注入函数 (核心黑科技)
# Pic2Word 的 model.py 中有 encode_text_img_retrieval，我们需要适配 OpenAI CLIP
def encode_text_with_token(clip_model, text_tokens, image_token_emb, token_idx_marker):
    """
    将 image_token_emb 插入到 text_tokens 中指定的位置 (token_idx_marker)。
    """
    dtype = clip_model.dtype

    # [Fix] Remove torch.no_grad() to allow backprop to mapper
    # A. 获取原始文本 Embedding [Batch, 77, Dim]
    x = clip_model.token_embedding(text_tokens).type(dtype)

    # B. 替换 Token
    # 找到标记符 '*' 的位置
    mask = (text_tokens == token_idx_marker) # [Batch, 77]

    if mask.any():
        # 将映射后的图像 Token 注入
        # image_token_emb: [Batch, Dim] -> [Batch, 1, Dim]
        # 注意：这里假设每个句子只有一个 '*'
        for i in range(x.shape[0]):
            idx = (text_tokens[i] == token_idx_marker).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                x[i, idx[0], :] = image_token_emb[i].type(dtype)

    # C.1. 处理 Positional Embedding (区分 CLIP vs Long-CLIP)
    # Long-CLIP has attributes 'positional_embedding_res', 'mask1', 'mask2'
    if hasattr(clip_model, 'positional_embedding_res'):
        # Long-CLIP Logic
        # x: [Batch, SeqLen, Dim]
        # masks: [SeqLen, 1]

        pos_emb = clip_model.positional_embedding.to(x.device)
        pos_emb_res = clip_model.positional_embedding_res.to(x.device)
        mask1 = clip_model.mask1.to(x.device)
        mask2 = clip_model.mask2.to(x.device)

        # Apply PE
        # Note: Long-CLIP model code does x.permute(1,0,2) AFTER PE application in encode_text.
        # But we are injecting tokens BEFORE PE ? No, 'token_embedding' is applied first.
        # LongCLIP.encode_text:
        #   x = token_emb(text)
        #   x = x + (pos * m1) + (pos_res * m2)
        #   x = transformer(x)

        pe_term = (pos_emb * mask1).type(dtype) + (pos_emb_res * mask2).type(dtype)
        x = x + pe_term
    else:
        # Standard CLIP Logic
        x = x + clip_model.positional_embedding.type(dtype)

    # C.2. Transformer Forward
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(dtype)

    # D. 提取 EOT 特征
    # text_tokens.argmax(dim=-1) 找到 EOT Token 的索引
    # [Fix] Ensure indices are long
    eot_indices = text_tokens.argmax(dim=-1).long()

    # [Fix] Gather EOT features carefully to maintain gradient
    # x is [Briefly Batch, SeqLen, Dim]
    # We want [Batch, Dim] at eot_indices
    # x[torch.arange(x.shape[0]), eot_indices] is correct
    x = x[torch.arange(x.shape[0]), eot_indices] @ clip_model.text_projection

    return x
