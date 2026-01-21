import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import clip
import sys
from PIL import Image
from torchvision import transforms

# Add Long-CLIP to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Long-CLIP'))
try:
    from model import longclip
except ImportError:
    print("Error: Long-CLIP not found in path.")

from pic2word import IM2TEXT, encode_text_with_token

class AircraftSelfSumDataset(Dataset):
    """
    Self-Supervised Dataset for Aircraft.
    Task: RefImage -> (Mapper) -> "a photo of *" -> Match TargetImage (Same Image / Augmented)
    """
    def __init__(self, root_dir, preprocess):
        self.root_dir = root_dir
        self.image_paths = []

        # Load from images folder
        img_dir = os.path.join(root_dir, "data/images")
        # Try different list files or fallback to iteration
        list_files = [
            os.path.join(root_dir, "data/images_train.txt"),
            os.path.join(root_dir, "data/images_variant_train.txt")
        ]

        found_list = False
        for list_file in list_files:
            if os.path.exists(list_file):
                print(f"Loading image list from {list_file}")
                with open(list_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        # format might be "ID variant" or just "ID"
                        parts = line.split(" ")
                        img_id = parts[0]
                        path = os.path.join(img_dir, f"{img_id}.jpg")
                        if os.path.exists(path):
                            self.image_paths.append(path)
                found_list = True
                break

        if not found_list:
            # Fallback to walk
            print(f"No list file found, walking directory {img_dir}")
            for root, dirs, files in os.walk(img_dir):
                for file in files:
                    if file.endswith(".jpg"):
                         self.image_paths.append(os.path.join(root, file))

        print(f"Loaded {len(self.image_paths)} images from {img_dir}")
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img_pil = Image.open(path).convert("RGB")

            # Simple Self-Supervised task: Identity Mapping
            # In a better setup, we would use different augmentations for ref and target
            # Due to preprocess being fixed (resize/crop), they are mostly identical here

            ref_image = self.preprocess(img_pil)
            target_image = ref_image.clone()

            # Simpler Prompt
            text = "a photo of *"

            return ref_image, text, target_image
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return dummy zero tensors on error
            return torch.zeros(3, 224, 224), "error", torch.zeros(3, 224, 224)

# Keep Dummy just in case
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, length=100): self.length = length
    def __len__(self): return self.length
    def __getitem__(self, idx): return torch.randn(3,224,224), "test", torch.randn(3,224,224)

def train(args):
    device = args.device
    if device.isdigit():
        device = f"cuda:{device}"

    # 1. Load Long-CLIP
    print(f"Loading Long-CLIP from {args.longclip_path}...")
    model, preprocess = longclip.load(args.longclip_path, device=device)
    model.eval()

    # Freeze CLIP
    for param in model.parameters():
        param.requires_grad = False

    # 2. Initialize Mapper
    print("Initializing Pic2Word Mapper...")
    embed_dim = model.visual.output_dim
    print(f"Detected Embedding Dimension: {embed_dim}")

    mapper = IM2TEXT(embed_dim=embed_dim, output_dim=model.token_embedding.weight.shape[1]).to(device)
    mapper.train()

    # 3. Optimizer
    optimizer = optim.AdamW(mapper.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 4. Data Loader
    # Use real dataset
    aircraft_root = "datasets/fgvc-aircraft-2013b"

    if os.path.exists(aircraft_root):
        print("Using Real Aircraft Dataset")
        dataset = AircraftSelfSumDataset(aircraft_root, preprocess)
    else:
        print(f"Warning: Dataset not found at {aircraft_root}, using dummy.")
        dataset = DummyDataset()

    if len(dataset) == 0:
        print("Dataset empty, using dummy.")
        dataset = DummyDataset()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    token_idx_marker = longclip.tokenize(["*"])[0][1] # Get token ID for '*'

    print(f"Starting Training on {len(dataset)} images...")
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(dataloader)

        for batch in pbar:
            ref_images, texts, target_images = batch

            # Simple error handling
            if "error" in texts: continue
            if isinstance(texts, tuple): texts = list(texts)

            ref_images = ref_images.to(device)
            target_images = target_images.to(device)
            bs = ref_images.size(0)

            optimizer.zero_grad()

            # Forward Logic

            # 1. Encode Ref Image -> Token
            with torch.no_grad():
                ref_img_feat = model.encode_image(ref_images)
                ref_img_feat = ref_img_feat / ref_img_feat.norm(dim=-1, keepdim=True)

            # Map to Token Space (Gradients start here)
            # Cast to float32 to match mapper dtype
            ref_token = mapper(ref_img_feat.detach().float()) # [Batch, Dim]

            # 2. Encode Composed Text
            text_tokens = longclip.tokenize(texts).to(device)

            # Inject Token and Encode text
            composed_text_feat = encode_text_with_token(model, text_tokens, ref_token, token_idx_marker)
            composed_text_feat = composed_text_feat / composed_text_feat.norm(dim=-1, keepdim=True)

            # 3. Encode Target Image
            with torch.no_grad():
                target_img_feat = model.encode_image(target_images)
                target_img_feat = target_img_feat / target_img_feat.norm(dim=-1, keepdim=True)

            # 4. Contrastive Loss
            # (In-batch negatives)
            logits = (composed_text_feat @ target_img_feat.T) * model.logit_scale.exp()
            labels = torch.arange(bs).to(device)

            loss = nn.CrossEntropyLoss()(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_description(f"Ep {epoch} Loss: {loss.item():.4f}")

        # Save Checkpoint
        save_path = f"pic2word_longclip_ep{epoch}.pt"
        torch.save(mapper.state_dict(), save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--longclip_path", type=str, default="Long-CLIP/checkpoints/longclip-L.pt")

    args = parser.parse_args()
    train(args)