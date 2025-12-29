import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Dataset Implementations
# -----------------------------------------------------------------------------

class AircraftDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, "data")
        self.images_dir = os.path.join(self.data_dir, "images")

        list_file = os.path.join(self.data_dir, f"images_variant_{split}.txt")
        self.samples = []

        with open(list_file, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    img_id, variant = parts
                    self.samples.append({
                        "id": img_id,
                        "label": variant,
                        "path": os.path.join(self.images_dir, f"{img_id}.jpg")
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        try:
            image = Image.open(item["path"]).convert("RGB")
        except Exception as e:
            print(f"Error loading {item['path']}: {e}")
            # Return a dummy image or handle error
            image = Image.new("RGB", (224, 224))

        # Construct prompt
        prompt = f"a photo of a {item['label']}"

        return {
            "image": image,
            "text": prompt,
            "path": item["path"]
        }

class CUBDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "images")

        # Load classes
        self.classes = {}
        with open(os.path.join(root_dir, "classes.txt"), "r") as f:
            for line in f:
                cid, cname = line.strip().split(" ", 1)
                # Clean class name: 001.Black_footed_Albatross -> Black footed Albatross
                clean_name = cname.split(".", 1)[1].replace("_", " ")
                self.classes[cid] = clean_name

        # Load image labels
        self.image_labels = {}
        with open(os.path.join(root_dir, "image_class_labels.txt"), "r") as f:
            for line in f:
                iid, cid = line.strip().split(" ", 1)
                self.image_labels[iid] = cid

        # Load images and split
        self.samples = []

        # CUB doesn't have a standard train/test split file in the root usually,
        # but often uses train_test_split.txt
        split_map = {}
        with open(os.path.join(root_dir, "train_test_split.txt"), "r") as f:
            for line in f:
                iid, is_train = line.strip().split(" ", 1)
                split_map[iid] = int(is_train)

        target_split = 1 if split == "train" else 0

        with open(os.path.join(root_dir, "images.txt"), "r") as f:
            for line in f:
                iid, path = line.strip().split(" ", 1)
                if split_map.get(iid) == target_split:
                    cid = self.image_labels[iid]
                    cname = self.classes[cid]
                    self.samples.append({
                        "id": iid,
                        "label": cname,
                        "path": os.path.join(self.images_dir, path)
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        try:
            image = Image.open(item["path"]).convert("RGB")
        except Exception as e:
            print(f"Error loading {item['path']}: {e}")
            image = Image.new("RGB", (224, 224))

        prompt = f"a photo of a {item['label']}"

        return {
            "image": image,
            "text": prompt,
            "path": item["path"]
        }

# -----------------------------------------------------------------------------
# Model Wrapper & Loss
# -----------------------------------------------------------------------------

class VLM2VecWrapper(nn.Module):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor

    def forward(self, images, texts):
        # 1. Encode Texts (Queries)
        # Instruction: "Retrieve the image that matches the description: {text}"
        text_messages = [
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Retrieve the image that matches the description: {t}"}]
                }
            ] for t in texts
        ]

        text_inputs = self.processor.apply_chat_template(
            text_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        text_outputs = self.model(**text_inputs, output_hidden_states=True)
        # Extract last token of last layer
        # text_inputs['input_ids'] shape: [B, Seq]
        # We need to find the index of the last token (usually just -1, but padding might exist)
        # Since we use padding=True, we should use attention_mask to find the last real token

        # Simple approach: use -1 if left-padding is used (common for generation),
        # but Qwen processor might use right padding.
        # Let's check attention mask.

        # Assuming right padding for now as is common with HF processors unless specified
        # Actually, for causal LM, left padding is often preferred for generation, but here we just encode.
        # Let's use the attention mask to find the last token index.

        last_token_indices = text_inputs.attention_mask.sum(dim=1) - 1

        # hidden_states is a tuple, take the last one
        last_hidden = text_outputs.hidden_states[-1] # [B, Seq, Dim]

        # Gather the last token embeddings
        text_embeddings = last_hidden[torch.arange(last_hidden.size(0)), last_token_indices]

        # 2. Encode Images (Documents)
        # Instruction: Image + "Represent this image for retrieval."
        image_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Represent this image for retrieval."}
                    ]
                }
            ] for img in images
        ]

        # Processor handles images automatically
        # Note: apply_chat_template with images usually returns text prompt,
        # we need to call processor(text=..., images=...)

        # First, generate text prompts for all images
        image_text_prompts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in image_messages
        ]

        image_inputs = self.processor(
            text=image_text_prompts,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        image_outputs = self.model(**image_inputs, output_hidden_states=True)

        # Find last token for images
        # Note: Qwen2-VL handles images as tokens in the sequence.
        last_img_token_indices = image_inputs.attention_mask.sum(dim=1) - 1
        last_img_hidden = image_outputs.hidden_states[-1]
        image_embeddings = last_img_hidden[torch.arange(last_img_hidden.size(0)), last_img_token_indices]

        return text_embeddings, image_embeddings

def info_nce_loss(text_embeddings, image_embeddings, temperature=0.05):
    # Normalize
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

    # Similarity Matrix [B, B]
    logits = torch.matmul(text_embeddings, image_embeddings.T) / temperature

    # Labels: 0, 1, 2, ... B-1
    labels = torch.arange(logits.size(0), device=logits.device)

    # Symmetric Loss
    loss_t2i = F.cross_entropy(logits, labels)
    loss_i2t = F.cross_entropy(logits.T, labels)

    return (loss_t2i + loss_i2t) / 2

# -----------------------------------------------------------------------------
# Main Training Script
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL for Retrieval (VLM2Vec)")
    parser.add_argument("--dataset", type=str, required=True, choices=["aircraft", "cub"], help="Dataset name")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model path")
    parser.add_argument("--output_dir", type=str, default="output/vlm2vec_finetuned", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device ID (e.g. '0', 'cuda:1')")
    parser.add_argument("--disable_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing (enabled by default to save memory)")

    args = parser.parse_args()

    # Handle numeric device ID
    if args.device.isdigit():
        args.device = f"cuda:{args.device}"

    # 1. Load Dataset
    print(f"Loading {args.dataset} dataset from {args.dataset_root}...")
    if args.dataset == "aircraft":
        train_dataset = AircraftDataset(args.dataset_root, split="train")
    elif args.dataset == "cub":
        train_dataset = CUBDataset(args.dataset_root, split="train")

    print(f"Found {len(train_dataset)} training samples.")

    # Collate function to handle list of dicts
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]
        return images, texts

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    # 2. Load Model
    print(f"Loading model from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )

    # Enable gradient checkpointing by default to save memory
    if not args.disable_gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        # Required for PEFT with gradient checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 3. Apply LoRA
    print("Applying LoRA...")
    # Target modules for Qwen2-VL (Language Model parts usually sufficient for VLM2Vec)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, # Qwen2-VL is technically CausalLM structure
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    wrapper = VLM2VecWrapper(model, processor)

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 5. Training Loop
    print("Starting training...")
    model.train()

    os.makedirs(args.output_dir, exist_ok=True)

    all_losses = []

    for epoch in range(args.epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, (images, texts) in enumerate(progress_bar):
            optimizer.zero_grad()

            # Forward
            # Note: images is list of PIL Images, texts is list of strings
            text_embs, img_embs = wrapper(images, texts)

            # Loss
            loss = info_nce_loss(text_embs, img_embs, temperature=args.temperature)

            # Backward
            loss.backward()
            optimizer.step()

            all_losses.append(loss.item())
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

            if (step + 1) % 100 == 0:
                # Save checkpoint
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-ep{epoch}-step{step}")
                model.save_pretrained(ckpt_dir)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

        # Save epoch checkpoint
        model.save_pretrained(os.path.join(args.output_dir, f"epoch_{epoch+1}"))

    print("Training finished.")
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(all_losses, label="Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))
    print(f"Loss curve saved to {os.path.join(args.output_dir, 'loss_curve.png')}")
    print(f"Model saved to {os.path.join(args.output_dir, 'final_model')}")

if __name__ == "__main__":
    main()
