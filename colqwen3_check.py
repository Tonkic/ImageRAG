import torch
from transformers import AutoModel, AutoProcessor, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from PIL import Image
import requests
import gc

# ================= 配置部分 =================
LOCAL_MODEL_PATH = "TomoroAI/tomoro-colqwen3-embed-8b"
TARGET_DEVICE_ID = 3  # 您指定的显卡 ID

# 构造设备字符串
DEVICE_STR = f"cuda:{TARGET_DEVICE_ID}"

# ================= 环境清理与准备 =================
# 1. 清理显存，防止之前的报错残留
gc.collect()
torch.cuda.empty_cache()

print(f"🚀 启动高性能模式 (Native BF16)")
print(f"🎯 使用设备: {DEVICE_STR} (RTX 3090)")

# ================= 核心修复函数 (万能 Patch) =================
def load_model_high_performance(model_path, device_str):
    """
    以原生 BF16 加载模型，并修复所有参数兼容性问题。
    """
    print("🛠️  正在 Patch 模型代码以适配新版 Transformers...")

    # 1. 获取模型类定义
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    class_ref = config.auto_map["AutoModel"]
    model_class = get_class_from_dynamic_module(class_ref, model_path)

    # 2. 应用万能补丁 (吞掉所有不认识的参数)
    if hasattr(model_class, "tie_weights"):
        original_tie_weights = model_class.tie_weights

        def safe_tie_weights(self, **kwargs):
            # 这里的 **kwargs 会捕获 missing_keys, recompute_mapping 等所有参数
            # 我们不传给原函数，直接丢弃，从而避免 TypeError
            return original_tie_weights(self)

        model_class.tie_weights = safe_tie_weights
        print("✅ tie_weights Patch 成功 (已屏蔽所有未知参数)")

    # 3. 加载模型 (BF16 + Flash Attention 2)
    print(f"🔥 正在加载完整模型 (BF16)... 这需要约 16GB 显存")
    model = model_class.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,        # 关键：使用原生 BF16
        attn_implementation="flash_attention_2", # 关键：开启加速
        trust_remote_code=True,
        device_map={"": device_str}        # 强制指定单卡
    ).eval()

    return model

# ================= 主流程 =================

# 1. 加载处理器
processor = AutoProcessor.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True,
    max_num_visual_tokens=1280,
)

# 2. 加载模型
try:
    model = load_model_high_performance(LOCAL_MODEL_PATH, DEVICE_STR)
    print(f"✨ 模型加载成功！显存占用正常。")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit()

# ================= 准备数据 =================
queries = [
    "Retrieve the city of Singapore",
    "Retrieve the city of Beijing",
]

def load_image(url):
    try:
        response = requests.get(url, stream=True, timeout=10)
        return Image.open(response.raw).convert("RGB")
    except Exception:
        print(f"无法下载图片: {url}")
        return Image.new('RGB', (224, 224), color='gray')

print("📥 正在加载图片...")
image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Singapore_skyline_2022.jpg/640px-Singapore_skyline_2022.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/Beijing_skyline_at_night.JPG/640px-Beijing_skyline_at_night.JPG"
]
images = [load_image(url) for url in image_urls]

# ================= 推理 (高性能模式) =================
print("🧠 开始推理 (BF16 Precision)...")

# 文本编码
batch_queries = processor.process_texts(queries)
batch_queries = {k: v.to(DEVICE_STR) for k, v in batch_queries.items()}

with torch.inference_mode():
    query_outputs = model(**batch_queries)
    # 保持在 GPU 上进行打分计算会更快，最后再转 CPU
    query_embeddings = query_outputs.embeddings

# 图片编码
batch_images = processor.process_images(images)
batch_images = {k: v.to(DEVICE_STR) for k, v in batch_images.items()}

with torch.inference_mode():
    image_outputs = model(**batch_images)
    doc_embeddings = image_outputs.embeddings

# ================= 打分 =================
# 注意：score_multi_vector 可能会在 CPU 上运行，我们需要确保 tensor 在 CPU
# 或者如果库支持 GPU 计算，可以尝试不转。为了稳妥，这里转回 CPU。
scores = processor.score_multi_vector(
    query_embeddings.to(torch.float32).cpu(),
    doc_embeddings.to(torch.float32).cpu()
)

print("\n=== 检索结果 ===")
for i, query in enumerate(queries):
    print(f"\n🔍 查询: '{query}'")
    for j, url in enumerate(image_urls):
        print(f"   -> 图片 {j+1} 分数: {scores[i][j].item():.4f}")