import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

from transformers.utils.import_utils import is_flash_attn_2_available

class MemoryProjector(nn.Module):
    """
    MemoRAG-inspired Compact Memory Module.
    Compresses variable-length image sequences into fixed-size memory tokens,
    then allows text to query this compressed memory.
    """
    def __init__(self, input_dim=128, hidden_dim=256, num_memory_tokens=16, num_heads=4):
        super().__init__()

        # [MemoRAG] Learnable Memory Tokens (Latent Queries)
        # Compresses N image patches into K memory tokens
        self.memory_tokens = nn.Parameter(torch.randn(1, num_memory_tokens, input_dim))

        # Compressor: Memory Tokens attend to Image Patches
        # Query=Memory, Key=Image, Value=Image
        self.compressor = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

        # Retrieval: Text attends to Compressed Memory
        # Query=Text, Key=Memory, Value=Memory
        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

        # Scoring Head
        self.score_head = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, img_seq, text_seq):
        # img_seq: [Batch, Seq_img, Dim]
        # text_seq: [Batch, Seq_txt, Dim]
        batch_size = img_seq.shape[0]

        # 1. Memory Formation (Compression)
        # Expand memory tokens for batch
        mem_query = self.memory_tokens.repeat(batch_size, 1, 1) # [B, K, D]

        # Q=Memory, K=Image, V=Image
        compressed_mem, _ = self.compressor(query=mem_query, key=img_seq, value=img_seq)
        # compressed_mem: [B, K, D]

        # 2. Retrieval (Text queries Memory)
        # Q=Text, K=Memory, V=Memory
        attn_out, _ = self.cross_attn(query=text_seq, key=compressed_mem, value=compressed_mem)
        # attn_out: [B, Seq_txt, D]

        # 3. Aggregation
        # Mean pool text context and original text
        txt_context = attn_out.mean(dim=1) # [B, D]
        txt_original = text_seq.mean(dim=1) # [B, D]

        combined = torch.cat([txt_original, txt_context], dim=-1) # [B, 2*D]

        return self.score_head(combined)

class GlobalMemory:
    def __init__(self, memory_file="global_memory.json", model_path="global_memory_model.pth", device="cuda", embedding_model="colqwen2.5", adapter_path=None):
        self.memory_file = memory_file
        self.model_path = model_path
        self.device = device
        self.embedding_model_type = embedding_model
        self.adapter_path = adapter_path

        # self.memory = self._load_memory()
        self.memory = [] # Always start fresh to ensure experiment independence

        # Initialize history for Taboo Search
        self.history = set()

        # [Lazy Loading] Do not load models immediately to save VRAM
        self.model = None
        self.processor = None

        # Projector will be initialized after model loading when dim is known
        self.projector = None

        # Default model names
        self.qwen_name = "Qwen/Qwen2.5-VL-3B-Instruct"

        print(f"Initialized new Global Memory Model (In-Memory Only). Embedding: {self.embedding_model_type}")

    def _load_model(self):
        """Lazy load the appropriate embedding model."""
        if self.model is not None:
            return

        if self.embedding_model_type == "Qwen2.5-VL":
            self._load_qwen2_5_vl()
        else:
            print(f"Warning: Unknown embedding model type '{self.embedding_model_type}'. Defaulting to Qwen2.5-VL.")
            self._load_qwen2_5_vl()

    def _load_qwen2_5_vl(self):
        print(f"Loading Qwen2.5-VL for Global Memory (Adapter: {self.adapter_path})...")
        try:
            # Use local path if available
            local_path = "models/Qwen/Qwen2.5-VL-3B-Instruct"
            model_name = local_path if os.path.exists(local_path) else self.qwen_name

            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True
            ).eval()

            if self.adapter_path:
                try:
                    from peft import PeftModel
                    print(f"Loading LoRA adapter from {self.adapter_path}...")
                    self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
                except ImportError:
                    print("Warning: peft not installed, cannot load adapter.")
                except Exception as e:
                    print(f"Error loading adapter: {e}")

            # Determine hidden size from config
            if hasattr(self.model.config, "hidden_size"):
                hidden_size = self.model.config.hidden_size
            elif hasattr(self.model.config, "text_config") and hasattr(self.model.config.text_config, "hidden_size"):
                hidden_size = self.model.config.text_config.hidden_size
            else:
                print("Warning: Could not determine hidden size from config, defaulting to 2048 (Qwen2.5-VL-3B).")
                hidden_size = 2048

            print(f"Qwen2.5-VL Hidden Size: {hidden_size}")

            # Initialize projector
            if self.projector is None:
                self.projector = MemoryProjector(input_dim=hidden_size).to(self.device)

        except Exception as e:
            print(f"Error loading Qwen2.5-VL: {e}")
            raise e

    def _load_memory(self):
        # Disabled for experiment independence
        return []

    def _save_memory(self):
        # Disabled for experiment independence
        pass

    def add(self, path):
        """Add path to history for Taboo Search."""
        self.history.add(path)

    def __contains__(self, path):
        return path in self.history

    def re_rank(self, paths, scores, prompt=None, alpha=0.5):
        """
        Re-rank paths/scores by penalizing items in history AND using Memory Model prediction.
        """
        new_scores = []

        # Pre-compute prompt embedding if prompt is provided
        prompt_emb = None
        if prompt:
            prompt_emb = self._get_sequence_embedding(prompt_text=prompt)
            if prompt_emb is not None:
                # Add batch dim: [1, Seq, D]
                prompt_emb = prompt_emb.unsqueeze(0)

        for path, original_score in zip(paths, scores):
            # 1. Taboo Search (History Penalty)
            if path in self.history:
                new_scores.append(-999.0)
                continue

            # 2. Memory Model Scoring
            memory_score = 0.5 # Neutral default
            if prompt_emb is not None:
                img_emb = self._get_sequence_embedding(image_path=path)
                if img_emb is not None:
                    img_emb = img_emb.unsqueeze(0) # [1, Seq, D]

                    # Ensure projector is loaded
                    if self.projector:
                        self.projector.eval()
                        with torch.no_grad():
                            # Ensure shapes match for batching (here batch=1)
                            memory_score = self.projector(img_emb, prompt_emb).item()

            # 3. Hybrid Scoring
            # Normalize original score if needed, assuming it's roughly 0-1 or similar scale
            # If original score is dot product, it might be > 1.
            # For simplicity, we just blend.
            final_score = (1 - alpha) * original_score + alpha * memory_score
            new_scores.append(final_score)

        # Re-sort descending
        combined = sorted(zip(paths, new_scores), key=lambda x: x[1], reverse=True)

        if combined:
            new_paths, new_scores = zip(*combined)
            return list(new_paths), list(new_scores)
        return [], []

    def add_feedback(self, image_path, prompt, actual_label=None, is_match=True):
        """
        Record feedback for RLGF training.
        """
        entry = {
            'image_path': image_path,
            'prompt': prompt,
            'actual_label': actual_label,
            'is_match': is_match,
            'score': 1.0 if is_match else 0.0
        }
        self.memory.append(entry)
        self.add(image_path)

    def _get_sequence_embedding(self, image_path=None, prompt_text=None):
        """
        Helper: Extract embeddings as SEQUENCES (No Pooling).
        Returns a float32 tensor of shape [Seq_Len, Dim].
        """
        self._load_model() # Ensure model is loaded

        with torch.no_grad():
            # --- Qwen2.5-VL Logic ---
            if image_path:
                try:
                    image = Image.open(image_path).convert("RGB")
                    # Simple prompt for image feature extraction
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": "Describe this image."}
                            ]
                        }
                    ]
                    inputs = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                    ).to(self.device)

                    outputs = self.model(**inputs, output_hidden_states=True)
                    # Return full sequence of last hidden state
                    return outputs.hidden_states[-1][0].float()
                except Exception as e:
                    print(f"Error embedding image {image_path}: {e}")
                    return None

            if prompt_text:
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": f"Retrieve the image that matches the description: {prompt_text}"}]
                        }
                    ]
                    inputs = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                    ).to(self.device)

                    outputs = self.model(**inputs, output_hidden_states=True)
                    return outputs.hidden_states[-1][0].float()
                except Exception as e:
                    print(f"Error embedding text '{prompt_text}': {e}")
                    return None

        return None

    def train_model(self, epochs=10, plot_path='global_memory_loss.png'):
        """
        Train the projector using Pairwise Ranking Loss (RLGF).
        """
        if not self.memory:
            print("No memory to train on.")
            return

        optimizer = optim.Adam(self.projector.parameters(), lr=1e-4)
        # [RLGF] MarginRankingLoss: Score(Pos) > Score(Neg) + Margin
        criterion = nn.MarginRankingLoss(margin=0.1)
        self.projector.train()

        # Group memory by prompt to find pairs
        prompt_groups = {}
        for entry in self.memory:
            p = entry['prompt']
            if p not in prompt_groups: prompt_groups[p] = []
            prompt_groups[p].append(entry)

        # Prepare pairs
        pairs = []
        print("Preparing training pairs (RLGF)...")

        for prompt, entries in tqdm(prompt_groups.items(), desc="Grouping Pairs"):
            # Sort by score (assuming 'score' field exists from TAC, or derive from is_match)
            # If 'score' is missing, use is_match (1.0 vs 0.0)
            # We prefer using fine-grained scores if available
            entries.sort(key=lambda x: x.get('score', 1.0 if x.get('is_match') else 0.0), reverse=True)

            if len(entries) < 2: continue

            # Simple strategy: Best vs Rest
            pos_entry = entries[0]
            pos_score = pos_entry.get('score', 1.0 if pos_entry.get('is_match') else 0.0)

            for neg_entry in entries[1:]:
                neg_score = neg_entry.get('score', 1.0 if neg_entry.get('is_match') else 0.0)

                # Only train if there is a meaningful margin
                if pos_score - neg_score > 0.5:
                    pairs.append((pos_entry, neg_entry, prompt))

        if not pairs:
            print("No valid training pairs found (need score difference > 0.5).")
            return

        print(f"Training Global Memory Model on {len(pairs)} pairs...")
        loss_history = []

        for ep in range(epochs):
            total_loss = 0
            # Shuffle pairs
            import random
            random.shuffle(pairs)

            # Process one by one (or small batches if we implement padding)
            # For simplicity and variable sequence lengths, we process one pair at a time (Batch=1)
            # Gradient accumulation could be used for stability

            optimizer.zero_grad()

            batch_loss = 0
            accumulation_steps = 8

            for i, (pos_entry, neg_entry, prompt) in enumerate(tqdm(pairs, desc=f"Epoch {ep+1}")):
                # Get Embeddings (Cached or Computed)
                # Note: In a real system, cache these. Here we re-compute (slow but simple)
                pos_img = self._get_sequence_embedding(image_path=pos_entry['image_path'])
                neg_img = self._get_sequence_embedding(image_path=neg_entry['image_path'])
                txt_emb = self._get_sequence_embedding(prompt_text=prompt)

                if pos_img is None or neg_img is None or txt_emb is None: continue

                # Add Batch Dim [1, Seq, D]
                pos_img = pos_img.unsqueeze(0)
                neg_img = neg_img.unsqueeze(0)
                txt_emb = txt_emb.unsqueeze(0)

                # Forward
                score_pos = self.projector(pos_img, txt_emb)
                score_neg = self.projector(neg_img, txt_emb)

                # Loss
                target = torch.ones(1, device=self.device)
                loss = criterion(score_pos, score_neg, target)

                loss = loss / accumulation_steps
                loss.backward()
                batch_loss += loss.item()

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += batch_loss
                    batch_loss = 0

            # Final step
            if batch_loss > 0:
                optimizer.step()
                optimizer.zero_grad()
                total_loss += batch_loss

            avg_loss = total_loss / (len(pairs) / accumulation_steps)
            loss_history.append(avg_loss)
            print(f"Epoch {ep+1}/{epochs}: Avg Loss {avg_loss:.4f}")

        print("Global Memory Model Updated (In-Memory).")

        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, epochs + 1), loss_history, marker='o')
            plt.title('Global Memory Pairwise Loss')
            plt.savefig(plot_path)
            plt.close()
        except: pass

    def predict_score(self, image_path, prompt):
        """
        Returns a score (0-1) indicating match probability.
        """
        self.projector.eval()

        img_emb = self._get_sequence_embedding(image_path=image_path)
        txt_emb = self._get_sequence_embedding(prompt_text=prompt)

        if img_emb is None or txt_emb is None:
            return 0.5

        with torch.no_grad():
            # Add batch dim
            img_emb = img_emb.unsqueeze(0)
            txt_emb = txt_emb.unsqueeze(0)
            score = self.projector(img_emb, txt_emb)
            return score.item()