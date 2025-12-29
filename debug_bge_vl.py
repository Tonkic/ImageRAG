
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "BAAI/BGE-VL-large"

print(f"Loading {MODEL_NAME}...")
try:
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()

    print("Model loaded.")

    # Check embedding size
    if hasattr(model, "text_model") and hasattr(model.text_model, "embeddings"):
        # CLIP structure usually
        embed_layer = model.text_model.embeddings.token_embedding
        print(f"Text Embedding Size: {embed_layer.weight.shape}")
        pos_embed = model.text_model.embeddings.position_embedding
        print(f"Position Embedding Size: {pos_embed.weight.shape}")
        print(f"Text Model Config: {model.text_model.config}")
    elif hasattr(model, "get_input_embeddings"):
        embed_layer = model.get_input_embeddings()
        print(f"Input Embedding Size: {embed_layer.weight.shape}")
    else:
        print("Could not find embedding layer easily.")
        print(model)

    print("Setting processor...")
    model.set_processor(MODEL_NAME)

    print("Model attributes:", dir(model))

    if hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
        print(f"Tokenizer found. Vocab Size: {len(tokenizer)}")
    elif hasattr(model, "processor") and hasattr(model.processor, "tokenizer"):
        tokenizer = model.processor.tokenizer
        print(f"Tokenizer found in processor. Vocab Size: {len(tokenizer)}")
    else:
        print("Tokenizer not found in model attributes.")
        # Try to load tokenizer manually to check
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Manual Tokenizer Vocab Size: {len(tokenizer)}")

    # Check position_ids values
    if hasattr(model.text_model.embeddings, "position_ids"):
        pos_ids = model.text_model.embeddings.position_ids
        print(f"Position IDs buffer shape: {pos_ids.shape}")
        print(f"Position IDs first 10 values: {pos_ids[0, :10]}")

        # Check if they are valid
        if pos_ids.max() >= 77:
            print("WARNING: Position IDs are invalid (>= 77). Fixing them...")
            new_pos_ids = torch.arange(77).expand((1, -1)).to(model.device)
            model.text_model.embeddings.position_ids = new_pos_ids
            print(f"Fixed Position IDs: {model.text_model.embeddings.position_ids[0, :10]}")

    text = ["test"]

    print("\n--- Test 1: padding=False ---")
    inputs1 = model.processor(text=text, return_tensors="pt")

    try:
        out1 = model.encode_text(inputs1)
        print("Encode1 success!")
        print(out1.shape)
    except Exception as e:
        print(f"Encode1 failed: {e}")



except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
