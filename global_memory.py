import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import clip
from PIL import Image

class MemoryProjector(nn.Module):
    """
    A lightweight MLP to learn the mapping between Image+Text pairs and their validity (Match/Mismatch).
    Input: Concatenated CLIP embeddings (Image + Text)
    Output: Probability of Match (0.0 - 1.0)
    """
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, img_emb, text_emb):
        # Concatenate features: [Batch, 512] + [Batch, 512] -> [Batch, 1024]
        x = torch.cat([img_emb, text_emb], dim=-1)
        return self.net(x)

class GlobalMemory:
    def __init__(self, memory_file="global_memory.json", model_path="global_memory_model.pth", device="cuda"):
        self.memory_file = memory_file
        self.model_path = model_path
        self.device = device
        self.memory = self._load_memory()

        # Load CLIP for feature extraction (Frozen)
        print("Loading CLIP for Global Memory...")
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()

        # Initialize Small Trainable Model
        self.projector = MemoryProjector().to(device)
        if os.path.exists(model_path):
            print(f"Loading Global Memory Model from {model_path}")
            self.projector.load_state_dict(torch.load(model_path))
        else:
            print("Initialized new Global Memory Model.")

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return []

    def _save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)

    def add_feedback(self, image_path, prompt, actual_label=None, is_match=True):
        """
        Record feedback from the VLM/Critic.

        Strategy:
        1. If Match (is_match=True):
           - Add Positive Sample: (Image, Prompt) -> Label 1
        2. If Mismatch (is_match=False):
           - Add Negative Sample: (Image, Prompt) -> Label 0
           - If actual_label is provided, Add Positive Sample: (Image, "a photo of {actual_label}") -> Label 1
        """
        # 1. Record the direct feedback (Positive or Negative)
        entry = {
            "image_path": image_path,
            "prompt": prompt,
            "actual_label": actual_label,
            "is_match": is_match,
            "timestamp": os.path.getmtime(image_path) if os.path.exists(image_path) else 0
        }
        self.memory.append(entry)

        # 2. If we have a correction (Mismatch + Actual Label), add the implicit positive
        # Example: Prompt="707-320", Image=ImgA, Match=False, Actual="707-200"
        # We add: (ImgA, "a photo of a 707-200") -> True (Label 1)
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

    def train_model(self, epochs=10):
        """
        Train the small projector model on the collected memory.
        Should be called periodically or after a batch of feedback.
        """
        if not self.memory:
            print("No memory to train on.")
            return

        optimizer = optim.Adam(self.projector.parameters(), lr=1e-4)
        criterion = nn.BCELoss()

        self.projector.train()

        # Prepare Batch
        img_embs = []
        txt_embs = []
        labels = []

        print("Preparing training data from memory...")
        valid_entries = 0
        for entry in self.memory:
            if not os.path.exists(entry['image_path']): continue

            try:
                # Image Emb
                img = self.preprocess(Image.open(entry['image_path']).convert("RGB")).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    img_emb = self.clip_model.encode_image(img).float()
                    img_emb /= img_emb.norm(dim=-1, keepdim=True)

                # Text Emb
                txt = clip.tokenize([entry['prompt']], truncate=True).to(self.device)
                with torch.no_grad():
                    txt_emb = self.clip_model.encode_text(txt).float()
                    txt_emb /= txt_emb.norm(dim=-1, keepdim=True)

                img_embs.append(img_emb)
                txt_embs.append(txt_emb)
                labels.append(1.0 if entry['is_match'] else 0.0)
                valid_entries += 1
            except Exception as e:
                # print(f"Error processing {entry}: {e}")
                continue

        if not img_embs:
            print("No valid images found in memory.")
            return

        img_embs = torch.cat(img_embs)
        txt_embs = torch.cat(txt_embs)
        labels = torch.tensor(labels, device=self.device).unsqueeze(1)

        print(f"Training Global Memory Model on {valid_entries} samples...")
        for ep in range(epochs):
            optimizer.zero_grad()
            preds = self.projector(img_embs, txt_embs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            if (ep+1) % 5 == 0:
                print(f"Epoch {ep+1}/{epochs}: Loss {loss.item():.4f}")

        torch.save(self.projector.state_dict(), self.model_path)
        print("Global Memory Model Updated.")

    def predict_score(self, image_path, prompt):
        """
        Returns a score (0-1) indicating how likely the image matches the prompt
        according to the global memory model.
        """
        self.projector.eval()
        try:
            img = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
            txt = clip.tokenize([prompt], truncate=True).to(self.device)

            with torch.no_grad():
                img_emb = self.clip_model.encode_image(img).float()
                img_emb /= img_emb.norm(dim=-1, keepdim=True)

                txt_emb = self.clip_model.encode_text(txt).float()
                txt_emb /= txt_emb.norm(dim=-1, keepdim=True)

                score = self.projector(img_emb, txt_emb)
                return score.item()
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5 # Default neutral
