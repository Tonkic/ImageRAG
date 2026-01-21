import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoProcessor, AutoModelForVision2Seq, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# -----------------------------------------------------------------------------
# Dataset Implementations
# -----------------------------------------------------------------------------

class AircraftDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, "data")
        self.images_dir = os.path.join(self.data_dir, "images")

        list_file = os.path.join(self.data_dir, f"images_variant_{split}.txt")
        family_file = os.path.join(self.data_dir, f"images_family_{split}.txt")

        self.samples = []
        self.groups = defaultdict(list) # For Hard Negative Sampling

        # Load Families
        id_to_family = {}
        if os.path.exists(family_file):
            with open(family_file, "r") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        id_to_family[parts[0]] = parts[1]

        with open(list_file, "r") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    img_id, variant = parts
                    family = id_to_family.get(img_id, "Unknown")

                    self.samples.append({
                        "id": img_id,
                        "label": variant,
                        "family": family,
                        "path": os.path.join(self.images_dir, f"{img_id}.jpg")
                    })

                    # Group by family for hard negative mining
                    self.groups[family].append(idx)

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

class HierarchicalBatchSampler(Sampler):
    """
    Samples batches where images belong to the same group (Family) to induce hard negatives.
    """
    def __init__(self, groups, batch_size, drop_last=False):
        self.groups = groups
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.keys = list(groups.keys())

    def __iter__(self):
        batch = []
        # Shuffle groups order
        random.shuffle(self.keys)

        for key in self.keys:
            indices = self.groups[key][:] # Copy
            random.shuffle(indices)

            for idx in indices:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return sum(len(g) for g in self.groups.values()) // self.batch_size
        else:
            return (sum(len(g) for g in self.groups.values()) + self.batch_size - 1) // self.batch_size

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

        # Attempt to access the backbone transformer to skip the LM head (saves memory)
        self.backbone = model
        # Unwrap PeftModel if present
        if hasattr(self.backbone, "base_model"):
            self.backbone = self.backbone.base_model
        # Unwrap ForConditionalGeneration if present (e.g. Qwen2_5_VLForConditionalGeneration -> Qwen2_5_VLModel)
        if hasattr(self.backbone, "model"):
            self.backbone = self.backbone.model
            print("VLM2VecWrapper: Successfully unwrapped model to backbone (skipping LM head).")
        else:
            print("VLM2VecWrapper: Could not unwrap model to backbone. LM head will be computed.")

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

        # Use backbone if available to skip LM head
        text_outputs = self.backbone(**text_inputs, output_hidden_states=True)

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
        if hasattr(text_outputs, "last_hidden_state"):
            last_hidden = text_outputs.last_hidden_state
        else:
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

        image_outputs = self.backbone(**image_inputs, output_hidden_states=True)

        # Find last token for images
        # Note: Qwen2-VL handles images as tokens in the sequence.
        last_img_token_indices = image_inputs.attention_mask.sum(dim=1) - 1

        if hasattr(image_outputs, "last_hidden_state"):
            last_img_hidden = image_outputs.last_hidden_state
        else:
            last_img_hidden = image_outputs.hidden_states[-1]

        image_embeddings = last_img_hidden[torch.arange(last_img_hidden.size(0)), last_img_token_indices]

        return text_embeddings, image_embeddings

def info_nce_loss(text_embeddings, image_embeddings, memory_bank=None, temperature=0.05):
    # Normalize
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

    # Similarity Matrix [B, B]
    logits_local = torch.matmul(text_embeddings, image_embeddings.T) / temperature

    if memory_bank is not None and len(memory_bank) > 0:
        # Memory Bank Negatives [B, M]
        # Ensure memory bank is on same device and normalized
        # We assume memory_bank is a list of tensors or a tensor
        if isinstance(memory_bank, list):
            mb_tensor = torch.stack(memory_bank).to(text_embeddings.device)
        else:
            mb_tensor = memory_bank.to(text_embeddings.device)

        logits_external = torch.matmul(text_embeddings, mb_tensor.T) / temperature
        logits = torch.cat([logits_local, logits_external], dim=1) # [B, B+M]
    else:
        logits = logits_local

    # Labels: 0, 1, 2, ... B-1
    labels = torch.arange(logits.size(0), device=logits.device)

    # Symmetric Loss
    # T2I: Text queries Image (Local + Memory Bank)
    loss_t2i = F.cross_entropy(logits, labels)

    # I2T: Image queries Text (Local only, as we don't have text memory bank)
    loss_i2t = F.cross_entropy(logits_local.T, labels)

    return (loss_t2i + loss_i2t) / 2

# -----------------------------------------------------------------------------
# Main Training Script
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL for Retrieval (VLM2Vec)")
    parser.add_argument("--dataset", type=str, required=True, choices=["aircraft", "cub"], help="Dataset name")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--model_path", type=str, default="/home/tingyu/imageRAG/Qwen3-VL-4B-Instruct", help="Model path")
    parser.add_argument("--output_dir", type=str, default="output/vlm2vec_finetuned", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device ID (e.g. '0', 'cuda:1')")
    parser.add_argument("--disable_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing (enabled by default to save memory)")
    parser.add_argument("--memory_bank_size", type=int, default=1024, help="Size of memory bank for negatives")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")

    args = parser.parse_args()

    # Handle numeric device ID
    if args.device.isdigit():
        args.device = f"cuda:{args.device}"

    # 1. Load Dataset
    print(f"Loading {args.dataset} dataset from {args.dataset_root}...")
    train_dataset = None
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

    # Use HierarchicalBatchSampler for Aircraft to induce Hard Negatives
    batch_sampler = None
    shuffle = True
    if args.dataset == "aircraft":
        print("Using HierarchicalBatchSampler for Hard Negative Mining (Same Family, Different Variant)...")
        batch_sampler = HierarchicalBatchSampler(train_dataset.groups, batch_size=args.batch_size, drop_last=True)
        shuffle = False # Sampler handles shuffling

    if batch_sampler:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=4
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
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

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 5. Training Loop
    print("Starting training...")
    model.train()

    os.makedirs(args.output_dir, exist_ok=True)

    all_losses = []
    memory_bank = [] # List of detached tensors

    for epoch in range(args.epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, (images, texts) in enumerate(progress_bar):
            # Forward
            # Note: images is list of PIL Images, texts is list of strings
            text_embs, img_embs = wrapper(images, texts)

            # Prepare Memory Bank Tensor
            mb_tensor = None
            if memory_bank:
                mb_tensor = torch.cat(memory_bank, dim=0)

            # Loss
            loss = info_nce_loss(text_embs, img_embs, memory_bank=mb_tensor, temperature=args.temperature)

            # Normalize loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps

            # Backward
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Update Memory Bank
            with torch.no_grad():
                # Normalize before storing? info_nce_loss normalizes inside.
                # But consistent normalization is good.
                # Let's store raw embeddings to be safe, normalize in loss.
                # Detach is crucial!
                current_batch_neg = img_embs.detach()
                memory_bank.append(current_batch_neg)

                # Maintain size
                current_size = sum(t.size(0) for t in memory_bank)
                while current_size > args.memory_bank_size:
                    removed = memory_bank.pop(0)
                    current_size -= removed.size(0)

            all_losses.append(loss.item())
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

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
