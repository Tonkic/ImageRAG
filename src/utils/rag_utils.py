import os
import time
import psutil
import threading
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import base64
import io
import copy
from PIL import Image

# --- Global Stats & Monitoring ---
RUN_STATS = {"input_tokens": 0, "output_tokens": 0}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ResourceMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.history = {'timestamp': [], 'cpu': [], 'ram': [], 'gpu': []}
        self.running = False
        self.thread = None

    def _loop(self):
        while self.running:
            self.history['timestamp'].append(time.time())
            self.history['cpu'].append(psutil.cpu_percent())
            self.history['ram'].append(psutil.virtual_memory().used / (1024**3)) # GB
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                current_gpus = []
                for i in range(device_count):
                    current_gpus.append(torch.cuda.memory_allocated(i) / (1024**3))
                self.history['gpu'].append(current_gpus)
            else:
                self.history['gpu'].append([0])
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()

    def save_plots(self, output_dir):
        if not self.history['timestamp']: return

        # Normalize time
        start_t = self.history['timestamp'][0]
        times = [t - start_t for t in self.history['timestamp']]

        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(times, self.history['cpu'], label='CPU Usage (%)', color='blue')
        plt.ylabel('CPU %')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(times, self.history['ram'], label='RAM Usage (GB)', color='green')
        plt.ylabel('GB')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        if self.history['gpu']:
            gpu_data = list(zip(*self.history['gpu'])) # Transpose to get list of traces
            for i, gpu_trace in enumerate(gpu_data):
                plt.plot(times, gpu_trace, label=f'GPU {i} Memory (GB)')
        plt.ylabel('GB')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'resource_usage.png'))
        plt.close()

class UsageTrackingClient:
    def __init__(self, client):
        self.client = client
        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, parent):
            self.parent = parent
            self.completions = self.Completions(parent)

        class Completions:
            def __init__(self, parent):
                self.parent = parent

            def create(self, *args, **kwargs):
                response = self.parent.client.chat.completions.create(*args, **kwargs)
                if hasattr(response, 'usage') and response.usage:
                    # Handle both object and dict access if necessary, usually object
                    try:
                        RUN_STATS['input_tokens'] += response.usage.prompt_tokens
                        RUN_STATS['output_tokens'] += response.usage.completion_tokens
                    except: pass
                return response

# --- Experience Library ---
class ExperienceLibrary:
    def __init__(self):
        self.global_rules = []   # 永久保留
        self.specific_rules = [] # 切换类别时清空

    def add_rule(self, rule_text, is_global=False):
        target_list = self.global_rules if is_global else self.specific_rules
        # 简单的去重
        if rule_text not in target_list:
            target_list.append(rule_text)

    def get_context_str(self):
        context = []
        if self.global_rules:
            context.append("GENERAL STRATEGIES (Apply to ALL tasks):")
            context.extend([f"- {r}" for r in self.global_rules])

        if self.specific_rules:
            context.append("\nVISUAL CONSTRAINTS (Apply ONLY to this subject):")
            context.extend([f"- {r}" for r in self.specific_rules])

        return "\n".join(context)

    def reset_specific(self):
        """只清空特定类别的经验，保留通用经验"""
        self.specific_rules = []

# --- Local Qwen3-VL Wrapper ---
class LocalQwen3VLWrapper:
    def __init__(self, model_path, device_map="auto", shared_model=None, shared_processor=None):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        import torch

        if shared_model is not None and shared_processor is not None:
             print(f"Loading LocalQwen3VLWrapper with SHARED model...")
             self.model = shared_model
             self.processor = shared_processor
        else:
             # Check if model is already loaded in global scope to avoid reloading
             import sys
             already_loaded_model = getattr(sys.modules.get("__main__"), "GLOBAL_QWEN_MODEL", None)
             already_loaded_processor = getattr(sys.modules.get("__main__"), "GLOBAL_QWEN_PROCESSOR", None)

             if already_loaded_model is not None and already_loaded_processor is not None:
                  print(f"Reuse GLOBAL_QWEN_MODEL from main module...")
                  self.model = already_loaded_model
                  self.processor = already_loaded_processor
             else:
                print(f"Loading local Qwen3-VL from {model_path} with device_map={device_map}...")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    device_map=device_map,
                    trust_remote_code=True
                )
                self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, parent):
            self.parent = parent
            self.completions = self.Completions(parent)

        class Completions:
            def __init__(self, parent):
                self.parent = parent

            def create(self, model, messages, temperature=0.35, **kwargs):
                # Convert OpenAI messages to Qwen messages
                qwen_messages = []
                for msg in messages:
                    content = msg['content']
                    new_content = []
                    if isinstance(content, str):
                        new_content.append({"type": "text", "text": content})
                    elif isinstance(content, list):
                        for item in content:
                            if item['type'] == 'text':
                                new_content.append({"type": "text", "text": item['text']})
                            elif item['type'] == 'image_url':
                                url = item['image_url']['url']
                                if url.startswith("data:image"):
                                    # Decode base64
                                    try:
                                        header, encoded = url.split(",", 1)
                                        data = base64.b64decode(encoded)
                                        image = Image.open(io.BytesIO(data))
                                        if image.mode != 'RGB':
                                            image = image.convert('RGB')
                                        new_content.append({"type": "image", "image": image})
                                    except Exception as e:
                                        print(f"Error decoding base64 image: {e}")
                                else:
                                    new_content.append({"type": "image", "image": url})

                    qwen_messages.append({
                        "role": msg['role'],
                        "content": new_content
                    })

                # Prepare inputs
                inputs = self.parent.processor.apply_chat_template(
                    qwen_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.parent.model.device)

                max_new_tokens = int(kwargs.pop("max_new_tokens", kwargs.pop("max_tokens", 1024)))
                top_p = float(kwargs.pop("top_p", 0.8))
                top_k = int(kwargs.pop("top_k", 20))
                repetition_penalty = float(kwargs.pop("repetition_penalty", 1.15))
                no_repeat_ngram_size = int(kwargs.pop("no_repeat_ngram_size", 4))

                do_sample = kwargs.pop("do_sample", temperature > 0)
                if temperature <= 0 and do_sample:
                    temperature = 0.01

                # Generation Config (single source of truth to avoid warning noise)
                generation_config = copy.deepcopy(getattr(self.parent.model, "generation_config", None))
                if generation_config is None:
                    raise RuntimeError("Model has no generation_config; cannot configure generation safely.")

                generation_config.max_new_tokens = max_new_tokens
                generation_config.repetition_penalty = repetition_penalty
                generation_config.no_repeat_ngram_size = no_repeat_ngram_size
                generation_config.do_sample = bool(do_sample)

                if bool(do_sample):
                    generation_config.temperature = temperature if temperature > 0 else 0.01
                    generation_config.top_p = top_p
                    generation_config.top_k = top_k
                else:
                    # Reset sampling knobs to neutral values when greedy decoding
                    generation_config.temperature = 1.0
                    generation_config.top_p = 1.0
                    generation_config.top_k = 50
                    if hasattr(generation_config, "typical_p"):
                        generation_config.typical_p = 1.0

                gen_kwargs = {
                    "generation_config": generation_config,
                }

                passthrough_generation_keys = [
                    "min_new_tokens",
                    "num_beams",
                    "length_penalty",
                    "early_stopping",
                    "num_return_sequences",
                    "use_cache",
                    "renormalize_logits",
                    "typical_p",
                ]
                for key in passthrough_generation_keys:
                    if key in kwargs:
                        setattr(generation_config, key, kwargs[key])

                with torch.no_grad():
                    generated_ids = self.parent.model.generate(**inputs, **gen_kwargs)

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.parent.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                # Mock Response
                class MockObject:
                    def __init__(self, **kwargs):
                        for k, v in kwargs.items():
                            setattr(self, k, v)

                # Calculate Usage
                prompt_tokens = len(inputs.input_ids[0])
                completion_tokens = len(generated_ids_trimmed[0])
                usage = MockObject(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )

                return MockObject(choices=[
                    MockObject(message=MockObject(content=output_text))
                ], usage=usage)
