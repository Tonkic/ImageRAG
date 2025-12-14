import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm


from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available

class MemoryProjector(nn.Module):
    """
    A Cross-Attention based module to learn the mapping between Image+Text pairs and their validity.
    Input: ColQwen embeddings (Pooled Image + Pooled Text)
    Dimension: 128 (ColQwen)
    Output: Probability of Match (0.0 - 1.0)
    """
    def __init__(self, input_dim=128, hidden_dim=256, num_heads=4):
        super().__init__()

        # Cross Attention: Query=Text, Key=Image, Value=Image
        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

        # MLP Head
        # Input: Text + Image + Attended_Text -> 128 * 3
        self.net = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Store attention weights for visualization
        self.last_attn_weights = None

    def forward(self, img_emb, text_emb):
        # Inputs: [Batch, 128]
        # Reshape to [Batch, 1, 128] for Attention
        img_seq = img_emb.unsqueeze(1)
        text_seq = text_emb.unsqueeze(1)

        # Cross Attention: Text attends to Image
        # Query: Text, Key: Image, Value: Image
        # attn_output: [Batch, 1, 128]
        # attn_weights: [Batch, 1, 1]
        attn_out, attn_weights = self.cross_attn(query=text_seq, key=img_seq, value=img_seq)

        self.last_attn_weights = attn_weights

        # Squeeze back to [Batch, 128]
        attn_out = attn_out.squeeze(1)

        # Concatenate: Original Text + Original Image + Attended Features
        x = torch.cat([text_emb, img_emb, attn_out], dim=-1)

        return self.net(x)

class GlobalMemory:
    def __init__(self, memory_file="global_memory.json", model_path="global_memory_model.pth", device="cuda"):
        self.memory_file = memory_file
        self.model_path = model_path
        self.device = device
        # self.memory = self._load_memory()
        self.memory = [] # Always start fresh to ensure experiment independence

        # Initialize history for Taboo Search
        self.history = set()
        # for entry in self.memory:
        #     if 'image_path' in entry:
        #         self.history.add(entry['image_path'])

        # [Lazy Loading] Do not load ColQwen2.5 immediately to save VRAM
        self.model = None
        self.processor = None
        self.model_name = "vidore/colqwen2.5-v0.2"

        # Initialize Small Trainable Model
        # MLP input dimension is 128 (ColQwen vector dim)
        self.projector = MemoryProjector(input_dim=128).to(device)

        # Always initialize new model to ensure experiment independence
        print("Initialized new Global Memory Model (In-Memory Only).")

    def _load_colqwen(self):
        """Lazy load ColQwen2.5 only when needed."""
        if self.model is not None:
            return

        print("Loading ColQwen2.5 for Global Memory...")
        try:
            self.processor = ColQwen2_5_Processor.from_pretrained(self.model_name)
            self.model = ColQwen2_5.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16, # ColQwen standard dtype
                device_map=self.device,
                attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            ).eval()
        except Exception as e:
            print(f"Error loading ColQwen2.5: {e}")
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

    def re_rank(self, paths, scores, penalty=100.0):
        """
        Re-rank paths/scores by penalizing items in history.
        """
        combined = list(zip(paths, scores))
        penalized = []
        for p, s in combined:
            if p in self.history:
                s -= penalty
            penalized.append((p, s))

        # Re-sort descending
        penalized.sort(key=lambda x: x[1], reverse=True)

        if penalized:
            new_paths, new_scores = zip(*penalized)
            return list(new_paths), list(new_scores)
        return [], []

    def add_feedback(self, image_path, prompt, actual_label=None, is_match=True):
        """
        Record feedback from the VLM/Critic.
        (Logic remains same as original)
        """
        # 1. Record the direct feedback
        entry = {
            "image_path": image_path,
            "prompt": prompt,
            "actual_label": actual_label,
            "is_match": is_match,
            "timestamp": os.path.getmtime(image_path) if os.path.exists(image_path) else 0
        }
        self.memory.append(entry)
        self.history.add(image_path) # Add to history for Taboo Search

        # 2. Implicit positive sample from correction
        if not is_match and actual_label:
            correction_prompt = f"a photo of a {actual_label}"
            self.memory.append({
                "image_path": image_path,
                "prompt": correction_prompt,
                "actual_label": actual_label,
                "is_match": True,
                "timestamp": os.path.getmtime(image_path) if os.path.exists(image_path) else 0
            })

        self._save_memory()

    def _get_pooled_embedding(self, image_path=None, prompt_text=None):
        """
        Helper: Extract ColQwen embeddings and apply Mean Pooling.
        Returns a float32 tensor of shape [1, 128] for the MLP.
        """
        self._load_colqwen() # Ensure model is loaded

        with torch.no_grad():
            if image_path:
                try:
                    image = Image.open(image_path).convert("RGB")
                    # ColQwen processing
                    batch = self.processor.process_images([image]).to(self.device)
                    # Inference (bfloat16)
                    # Output is a list of tensors (one per image), shape [N_patches, 128]
                    emb = self.model(**batch)

                    # [CORE LOGIC]: Mean Pooling -> [1, 128]
                    # We pool along the patch dimension (dim=0)
                    pooled = emb[0].mean(dim=0, keepdim=True).float()
                    return pooled
                except Exception as e:
                    print(f"Error embedding image {image_path}: {e}")
                    return None

            if prompt_text:
                try:
                    batch = self.processor.process_queries([prompt_text]).to(self.device)
                    # Output shape: [1, N_tokens, 128]
                    emb = self.model(**batch)

                    # [CORE LOGIC]: Mean Pooling -> [1, 128]
                    # Pool along token dimension (dim=1) since batch is dim 0
                    pooled = emb[0].mean(dim=0, keepdim=True).float()
                    return pooled
                except Exception as e:
                    print(f"Error embedding text '{prompt_text}': {e}")
                    return None

        return None

    def train_model(self, epochs=10, plot_path='global_memory_loss.png'):
        """
        Train the projector using pooled ColQwen embeddings.
        """
        if not self.memory:
            print("No memory to train on.")
            return

        optimizer = optim.Adam(self.projector.parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        self.projector.train()

        img_embs = []
        txt_embs = []
        labels = []

        print("Preparing training data from memory (ColQwen)...")
        valid_entries = 0

        # Use tqdm to show progress as ColQwen inference is heavier than CLIP
        for entry in tqdm(self.memory, desc="Encoding Memory"):
            if not os.path.exists(entry['image_path']): continue

            # Get pooled features
            img_emb = self._get_pooled_embedding(image_path=entry['image_path'])
            txt_emb = self._get_pooled_embedding(prompt_text=entry['prompt'])

            if img_emb is not None and txt_emb is not None:
                img_embs.append(img_emb)
                txt_embs.append(txt_emb)
                labels.append(1.0 if entry['is_match'] else 0.0)
                valid_entries += 1

        if not img_embs:
            print("No valid data found.")
            return

        # Stack into batch tensors: [Batch, 128]
        img_embs = torch.cat(img_embs)
        txt_embs = torch.cat(txt_embs)
        labels = torch.tensor(labels, device=self.device).unsqueeze(1)

        print(f"Training Global Memory Model on {valid_entries} samples...")
        loss_history = []
        for ep in range(epochs):
            optimizer.zero_grad()
            preds = self.projector(img_embs, txt_embs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            if (ep+1) % 5 == 0:
                print(f"Epoch {ep+1}/{epochs}: Loss {loss.item():.4f}")

        # torch.save(self.projector.state_dict(), self.model_path)
        print("Global Memory Model Updated (In-Memory).")

        try:
            import matplotlib.pyplot as plt

            # Plot 1: Loss Curve
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, epochs + 1), loss_history, marker='o', linestyle='-', color='b')
            plt.title('Global Memory MLP Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(plot_path)
            plt.close()
            print(f"Loss plot saved to {plot_path}")

            # Plot 2: Cross Attention Weights (if available)
            # We visualize the attention weights from the last batch of the last epoch
            if self.projector.last_attn_weights is not None:
                attn_weights = self.projector.last_attn_weights.detach().cpu().numpy()
                # attn_weights shape: [Batch, 1, 1] -> flatten to [Batch]
                attn_weights = attn_weights.flatten()

                attn_plot_path = plot_path.replace("loss.png", "attn_weights.png")
                plt.figure(figsize=(10, 5))
                plt.hist(attn_weights, bins=20, color='orange', alpha=0.7)
                plt.title('Distribution of Cross-Attention Weights (Last Batch)')
                plt.xlabel('Attention Weight')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.savefig(attn_plot_path)
                plt.close()
                print(f"Attention weights plot saved to {attn_plot_path}")

        except ImportError:
            print("matplotlib not found, skipping loss plot.")
        except Exception as e:
            print(f"Error plotting loss: {e}")

    def predict_score(self, image_path, prompt):
        """
        Returns a score (0-1) indicating match probability.
        """
        self.projector.eval()

        img_emb = self._get_pooled_embedding(image_path=image_path)
        txt_emb = self._get_pooled_embedding(prompt_text=prompt)

        if img_emb is None or txt_emb is None:
            return 0.5 # Default neutral if error

        with torch.no_grad():
            score = self.projector(img_emb, txt_emb)
            return score.item()