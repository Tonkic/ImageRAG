
import os
import argparse
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from huggingface_hub import hf_hub_download

# --- BiRefNet Utilities (Copied from SR script) ---
GLOBAL_BIREFNET = None
def load_birefnet(model_path, device):
    global GLOBAL_BIREFNET
    if GLOBAL_BIREFNET is None:
        print(f"Loading BiRefNet from {model_path}...")
        try:
            # 1. Load Config & Class
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model_class = get_class_from_dynamic_module(
                config.auto_map["AutoModelForImageSegmentation"],
                model_path
            )

            # 2. Instantiate on CPU
            model = model_class(config)

            # 3. Load Weights
            weight_path = None
            state_dict = None
            try:
                # Try safetensors
                weight_path = hf_hub_download(repo_id=model_path, filename="model.safetensors")
                from safetensors.torch import load_file
                state_dict = load_file(weight_path)
            except Exception:
                try:
                    # Fallback to pytorch_model.bin
                    weight_path = hf_hub_download(repo_id=model_path, filename="pytorch_model.bin")
                    state_dict = torch.load(weight_path, map_location="cpu")
                except Exception as e_load:
                    print(f"Could not find weights for BiRefNet: {e_load}")
                    raise e_load

            # 4. Apply Weights
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"BiRefNet Weights loaded: {msg}")

            # 5. Move to Device
            GLOBAL_BIREFNET = model
            GLOBAL_BIREFNET.to(device)
            GLOBAL_BIREFNET.eval()

        except Exception as e:
            print(f"Error loading BiRefNet: {e}")
            import traceback
            traceback.print_exc()
            GLOBAL_BIREFNET = None
    return GLOBAL_BIREFNET

def remove_background_batch(image_paths, output_dir, model, device):
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    os.makedirs(output_dir, exist_ok=True)

    for img_path in tqdm(image_paths, desc="Processing Images"):
        file_name = os.path.basename(img_path)
        out_path = os.path.join(output_dir, file_name)

        if os.path.exists(out_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size

            input_images = transform_image(image).unsqueeze(0).to(device)

            with torch.no_grad():
                preds = model(input_images)[-1].sigmoid().cpu()

            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize((w, h))

            # Apply Mask
            out_img = image.copy()
            out_img.putalpha(mask)

            # Composite on White Background
            final_image = Image.new("RGB", image.size, (255, 255, 255))
            final_image.paste(out_img, (0, 0), out_img)

            final_image.save(out_path)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess Dataset: Remove Backgrounds using BiRefNet")
    parser.add_argument("--dataset_root", type=str, default="datasets/fgvc-aircraft-2013b/data/images")
    parser.add_argument("--output_root", type=str, default="datasets/fgvc-aircraft-2013b/data/images_nobg")
    parser.add_argument("--birefnet_model_path", type=str, default="ZhengPeng7/BiRefNet")
    parser.add_argument("--device_id", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")

    # 1. Load Model
    model = load_birefnet(args.birefnet_model_path, device)
    if not model:
        print("Failed to load model. Exiting.")
        return

    # 2. Get Images
    image_files = [os.path.join(args.dataset_root, f) for f in os.listdir(args.dataset_root)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(image_files)} images.")

    # 3. Process
    remove_background_batch(image_files, args.output_root, model, device)

    print("Done.")

if __name__ == "__main__":
    main()
