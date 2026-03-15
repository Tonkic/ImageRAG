# 模型文档

> **[❗ 重要说明]**
> 本项目依赖极高显存的生成模型(如 OmniGen V2, FLUX)和多模态大模型。
> 请在**本地开发**管线逻辑和代码并完成**语法检查**。模型的下载与实际推理运行应在**服务器环境**进行。

本项目集成了多种模型，分为 **图像生成模型**、**视觉语言模型 (VLM)**、**检索模型** 和 **视觉特征/分割模型** 四大类。

---

## 一、图像生成模型

### 1. OmniGen V2

| 项目 | 详情 |
|------|------|
| **简称** | OmniGen2 |
| **类型** | 统一多模态生成模型 (DiT-based) |
| **本地路径** | `/home/tingyu/imageRAG/OmniGen2/` |
| **模型权重** | `/home/tingyu/imageRAG/OmniGen2/OmniGen2/` (或 `pretrained_models/`) |
| **GitHub** | [VectorSpaceLab/OmniGen2](https://github.com/VectorSpaceLab/OmniGen2) |
| **特点** | 支持文本-图像生成、参考图像引导生成、多种融合模式 |
| **VRAM 需求** | ~16-24GB (可开启 CPU offloading) |

#### 目录结构

```
OmniGen2/
├── omnigen2/                  # 核心代码
│   ├── pipeline.py            # OmniGen2Pipeline
│   ├── model.py               # 模型定义
│   └── ...
├── OmniGen2/                  # 默认模型权重目录
├── pretrained_models/         # 预训练模型
├── train.py                   # 训练脚本
├── inference.py               # 推理脚本
├── app.py                     # Gradio Demo
├── scripts/                   # 运行脚本
└── docs/                      # 文档
```

#### 使用方式

```python
# 标准使用 (在实验脚本中)
from OmniGen2.omnigen2.pipeline import OmniGen2Pipeline

pipe = OmniGen2Pipeline.from_pretrained("OmniGen2/OmniGen2")
pipe.to(omnigen_device)

# 生成图像
images = pipe(
    prompt="a photo of 737-300 aircraft",
    input_images=[ref_image_path],      # 参考图像
    height=512, width=512,
    image_guidance_scale=1.6,
    text_guidance_scale=2.5,
    seed=42
)
```

```python
# Late Fusion 模式（自定义管线）
from custom_pipeline import CustomOmniGen2DiTLateFusionPipeline

pipe = CustomOmniGen2DiTLateFusionPipeline.from_pretrained("OmniGen2/OmniGen2")
# AR 只接收文本，图像通过 VAE latent 直接注入 DiT
```

```python
# Early Fusion 模式
from custom_pipeline import CustomOmniGen2AREarlyFusionPipeline

pipe = CustomOmniGen2AREarlyFusionPipeline.from_pretrained("OmniGen2/OmniGen2")
# 图像被 MLLM 转为 token 序列影响 Hidden States
```

#### 命令行参数

```bash
```bash
python OmniGenV2_TAC_DINO_Importance_Aircraft.py \
    --omnigen2_path ./OmniGen2 \
    --omnigen2_model_path OmniGen2/OmniGen2 \
    --transformer_lora_path <可选LoRA路径> \
    --enable_offload \          # CPU offloading 节省 VRAM
    --enable_taylorseer \       # TaylorSeer 加速
    --height 512 --width 512 \
    --image_guidance_scale 1.6 \
    --text_guidance_scale 2.5 \
    --negative_prompt "blurry, low quality, text, watermark"
```

---

### 3. Z-Image-Turbo

| 项目 | 详情 |
|------|------|
| **类型** | 快速文本到图像生成模型 |
| **本地路径** | `/home/tingyu/imageRAG/Z-Image-Turbo/` |
| **特点** | 低延迟推理，1-4 步生成 |

#### 目录结构

```
Z-Image-Turbo/
├── model_index.json
├── processor/
├── scheduler/
├── text_encoder/
├── tokenizer/
├── transformer/
└── vae/
```

#### 使用方式

```python
# ZImageDemo.py 示例
from diffusers import ZImagePipeline

pipe = ZImagePipeline.from_pretrained("./Z-Image-Turbo")
image = pipe(prompt="a beautiful landscape", num_inference_steps=4).images[0]
```

```bash
python ZImageDemo.py
```

---

### 4. FLUX.1-dev

| 项目 | 详情 |
|------|------|
| **类型** | 高质量文本到图像生成模型 |
| **加载方式** | 通过 HuggingFace Hub / 本地缓存 |
| **使用脚本** | `test_flux_kontext.py` |

#### 使用方式

```bash
# FLUX 测试样例
python test_flux_kontext.py
```

---

## 二、视觉语言模型 (VLM)

### 1. Qwen3-VL-4B-Instruct

| 项目 | 详情 |
|------|------|
| **简称** | Qwen3-VL |
| **类型** | 多模态视觉语言模型 |
| **本地路径** | `/home/tingyu/imageRAG/Qwen3-VL-4B-Instruct/` |
| **参数量** | 4B（支持 4-bit 量化） |
| **用途** | TAC 评估、VAR 重排序、知识规格生成、输入解释 |

#### 使用方式

**本地加载（推荐）**：

```bash
python OmniGenV2_TAC_DINO_Importance_Aircraft.py \
    --use_local_model_weight \
    --local_model_weight_path /home/tingyu/imageRAG/Qwen3-VL-4B-Instruct \
    --qwen_4bit  # 可选 4-bit 量化
```

**通过 API 调用**（SiliconFlow）：

```bash
python OmniGenV2_TAC_DINO_Importance_Aircraft.py \
    --openai_api_key YOUR_KEY \
    --llm_model Qwen/Qwen3-VL-30B-A3B-Instruct
```

**代码中的封装**：

```python
from rag_utils import LocalQwen3VLWrapper

# 创建兼容 OpenAI API 接口的本地模型封装
model = LocalQwen3VLWrapper(
    model_path="/home/tingyu/imageRAG/Qwen3-VL-4B-Instruct",
    device_map="auto",
    load_in_4bit=True
)

# 像 OpenAI client 一样调用
response = model.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "描述这张图片"}]
)
```

#### 主要功能

1. **TAC 诊断** (`taxonomy_aware_critic.py`): 分层评分 (0-10)，支持抗 Token 截断的 JSON 自愈解析
2. **VAR 重排序**: 从 Top-K 候选中精细化排序
3. **知识规格生成**: 生成领域特定的视觉识别规格
4. **输入解释**: 提取 prompt 中的主体、属性、创意细节

#### 新增 Critic 用法

Qwen3-VL 现在还承担一组新 `critic` 模块的评估：

| Critic | 输入形式 | 主要用途 |
|------|------|------|
| `fine_grained_alignment_critic.py` | 单图 + prompt | 检查细粒度属性是否一致 |
| `identity_preservation_critic.py` | 参考图 + 生成图 + prompt | 检查是否仍保持同一机型/鸟类亚种 |
| `visual_realism_critic.py` | 单图 + prompt | 检查结构伪影、材质异常、物理不合理 |
| `overall_t2i_alignment_critic.py` | 单图 + prompt | 检查整体语义覆盖与偏移 |
| `multi_axis_critic.py` | 统一封装 | 一次返回多轴评估结果 |

推荐策略：

- `taxonomy_aware_critic.py`：适合生成修复链路，因为它会产出 `refined_prompt` 与 `retrieval_queries`
- `identity_preservation_critic.py`：适合作为 Aircraft / CUB 实验里的主保真 gate
- `multi_axis_critic.py`：适合做离线分析与诊断报告

---

### 2. Qwen2.5-VL / Qwen3-VL-30B-A3B

| 项目 | 详情 |
|------|------|
| **用途** | 通过 API 调用的更大规模 VLM |
| **调用方式** | SiliconFlow API（兼容 OpenAI 格式） |
| **默认模型名** | `Qwen/Qwen3-VL-30B-A3B-Instruct` |

```python
from rag_utils import UsageTrackingClient

client = UsageTrackingClient(
    api_key="YOUR_API_KEY",
    base_url="https://api.siliconflow.cn/v1"  # 或其他兼容端点
)
```

---

## 三、检索模型

### 1. CLIP (OpenAI)

| 项目 | 详情 |
|------|------|
| **全称** | Contrastive Language-Image Pre-Training |
| **本地路径** | `/home/tingyu/imageRAG/CLIP/` |
| **使用版本** | ViT-B/32 |
| **嵌入维度** | 512 |
| **Token 限制** | 77 tokens |
| **用途** | 基础图像-文本检索 |

#### 目录结构

```
CLIP/
└── CLIP/                      # OpenAI CLIP 源码
    ├── clip/
    │   ├── model.py
    │   ├── clip.py
    │   └── ...
    └── ...
```

#### 使用方式

```bash
# 使用 CLIP 作为检索方法
python OmniGenV2_TAC_DINO_Importance_Aircraft.py \
    --retrieval_method CLIP \
    --device_id 0
```

```python
# 代码中使用
import clip
model, preprocess = clip.load("ViT-B/32", device="cuda")
```

---

### 2. Long-CLIP

| 项目 | 详情 |
|------|------|
| **全称** | Long-CLIP: Unlocking the Long-Text Capability of CLIP |
| **本地路径** | `/home/tingyu/imageRAG/Long-CLIP/` |
| **检查点路径** | `/home/tingyu/imageRAG/Long-CLIP/checkpoints/longclip-L.pt` |
| **GitHub** | [beichenzbc/Long-CLIP](https://github.com/beichenzbc/Long-CLIP) |
| **使用版本** | ViT-L/14 |
| **嵌入维度** | 768 |
| **Token 限制** | 248 tokens（比标准 CLIP 的 77 大幅扩展） |
| **用途** | 长文本图像检索（推荐默认检索方法） |

#### 目录结构

```
Long-CLIP/
├── checkpoints/
│   └── longclip-L.pt          # 预训练权重
├── model/                     # 模型定义
├── open_clip_long/            # 长文本 CLIP 实现
├── train/                     # 训练代码
├── eval/                      # 评估代码
├── SDXL/                      # SDXL 集成
└── demo.py
```

#### 使用方式

```bash
# 使用 LongCLIP 作为检索方法（推荐）
python OmniGenV2_TAC_DINO_Importance_Aircraft.py \
    --retrieval_method LongCLIP \
    --device_id 0
```

```python
# 代码中加载
from model import longclip
model, preprocess = longclip.load("Long-CLIP/checkpoints/longclip-L.pt", device="cuda")
# 支持最多 248 tokens 的长文本编码
```

---

### 3. SigLIP / SigLIP2

| 项目 | 详情 |
|------|------|
| **SigLIP 版本** | `google/siglip-so400m-patch14-384` |
| **类型** | Sigmoid Loss 预训练的视觉-语言模型 |
| **加载方式** | 通过 HuggingFace `transformers` 自动下载 |
| **用途** | 图像-文本检索、评估指标 |

**SigLIP2 模型族**（注意维度不兼容，不可混用缓存）：

| 模型 ID | 嵌入维度 | 对应数据集缓存 | 备注 |
|---------|---------|--------------|------|
| `google/siglip2-base-patch16-224` | 768d | aircraft | 旧缓存，`.pt` 文件无 `model_name` 元数据 |
| `google/siglip2-so400m-patch16-naflex` | 1152d | cub、imagenet | `.pt` 文件含 `model_name` 元数据 |

> **自动对齐**：`ImageRetriever` 会读取嵌入缓存中存储的 `model_name` 字段，在加载时自动热切换 SigLIP2 查询编码器至匹配的模型，无需手动指定 `--siglip2_model_id`。混合多数据集场景（aircraft + cub + imagenet）中每个子检索器独立对齐。

#### 使用方式

```bash
# 使用 SigLIP 检索
python src/experiments/OmniGenV2_IPC_AR.py --retrieval_method SigLIP \
    --retrieval_datasets aircraft cub imagenet \
    --text_api_key YOUR_KEY --vl_api_key YOUR_KEY

# 使用 SigLIP2 检索（自动对齐模型，无需手动指定 --siglip2_model_id）
python src/experiments/OmniGenV2_IPC_AR.py --retrieval_method SigLIP2 \
    --retrieval_datasets aircraft cub imagenet \
    --text_api_key YOUR_KEY --vl_api_key YOUR_KEY

# 若需强制指定 SigLIP2 模型（比如重新预计算缓存时）
python src/experiments/OmniGenV2_IPC_AR.py --retrieval_method SigLIP2 \
    --siglip2_model_id google/siglip2-so400m-patch16-naflex \
    --retrieval_datasets cub imagenet \
    --text_api_key YOUR_KEY --vl_api_key YOUR_KEY
```

```python
# 代码中使用 (memory_guided_retrieval.py)
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
```

---

### 4. Qwen3-VL 检索模式

| 项目 | 详情 |
|------|------|
| **用途** | 使用 VLM 的视觉编码器进行图像嵌入 |
| **加载方式** | 本地权重或 HuggingFace |
| **特点** | 利用 VLM 更强的语义理解能力 |

```bash
python OmniGenV2_TAC_DINO_Importance_Aircraft.py --retrieval_method Qwen3-VL
```

---

### 5. FlagEmbedding

| 项目 | 详情 |
|------|------|
| **全称** | FlagEmbedding by BAAI |
| **本地路径** | `/home/tingyu/imageRAG/FlagEmbedding/` |
| **GitHub** | [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) |
| **用途** | 通用嵌入检索框架，支持多种检索模型 |

#### 目录结构

```
FlagEmbedding/
├── FlagEmbedding/             # 核心库
├── Tutorials/                 # 教程
├── examples/                  # 使用示例
├── research/                  # 研究代码
├── scripts/                   # 脚本
└── dataset/                   # 数据集工具
```

---


## 四、视觉特征/分割模型

### 1. DINOv3

| 项目 | 详情 |
|------|------|
| **全称** | DINOv3 (Self-supervised Vision Transformer) |
| **本地路径** | `/home/tingyu/imageRAG/dinov3/` |
| **权重路径** | `/home/tingyu/imageRAG/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` |
| **GitHub** | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) |
| **用途** | 图像保真度评估（DINOv3 Score） |
| **架构** | ViT-B/16 |

#### 目录结构

```
dinov3/
├── dinov3/                    # 源码
│   ├── models/
│   ├── utils/
│   └── ...
├── hubconf.py                 # PyTorch Hub 配置
└── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth  # 预训练权重
```

#### 使用方式

DINOv3 主要用于评估生成图像与参考图像之间的视觉保真度：

```python
# evaluate_all_recursive.py 中使用
import torch
model = torch.hub.load("dinov3/", "dinov3_vitb16", source="local")
model.load_state_dict(
    torch.load("dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
)
# 计算两张图像的 DINOv3 特征相似度
```

---

### 2. SAM2

| 项目 | 详情 |
|------|------|
| **全称** | Segment Anything Model 2 |
| **本地路径** | `/home/tingyu/imageRAG/sam2/` |
| **GitHub** | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |
| **用途** | 图像/视频分割（可用于前景提取、区域分析） |

#### 目录结构

```
sam2/
├── sam2/                      # 核心源码
├── checkpoints/               # 模型检查点
├── demo/                      # 演示
├── notebooks/                 # Jupyter 笔记本
├── training/                  # 训练代码
└── tools/                     # 工具
```

#### 使用方式

SAM2 可用于图像分割和目标提取，辅助生成评估或数据预处理：

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

model = build_sam2("sam2_hiera_l", "sam2/checkpoints/sam2_hiera_large.pt")
predictor = SAM2ImagePredictor(model)
```

---

## 五、模型使用总览

### 按功能分组

| 功能 | 模型 | 指定方式 |
|------|------|----------|
| **图像生成** | OmniGen V2 | `--omnigen2_path ./OmniGen2` |
| | Z-Image-Turbo | `ZImageDemo.py` |
| | FLUX.1-dev | `test_flux_kontext.py` 脚本 |
| **VLM 评估/诊断** | Qwen3-VL-4B | `--use_local_model_weight` |
| | Qwen3-VL-30B-A3B | `--llm_model Qwen/Qwen3-VL-30B-A3B-Instruct` |
| **图像检索** | CLIP ViT-B/32 | `--retrieval_method CLIP` |
| | Long-CLIP ViT-L/14 | `--retrieval_method LongCLIP` |
| | SigLIP | `--retrieval_method SigLIP` |
| | SigLIP2 | `--retrieval_method SigLIP2` |
| | Qwen3-VL | `--retrieval_method Qwen3-VL` |
| **图像评估** | DINOv3 | `evaluate_all_recursive.py` |
| | InceptionV3 (FID/IS) | `evaluate_all_recursive.py` |
| | SigLIP (SigLIP Score) | `evaluate_all_recursive.py` |
| **图像分割** | SAM2 | 辅助用途 |

### VRAM 管理策略

当 VLM 和生成模型在同一 GPU 上时，项目实现了自动 VRAM 管理：

```python
# 生成阶段：将 VLM 和检索模型移到 CPU
move_helpers_to_cpu()
# ... OmniGen2 生成 ...

# 评估/检索阶段：将 helper 模型移回 GPU
move_helpers_to_gpu(target_device)
# ... TAC 评估 + 检索 ...
```

也可使用双 GPU 分离：

```bash
python OmniGenV2_TAC_VAR_Aircraft.py \
    --device_id 0 \         # OmniGen2 GPU
    --vlm_device_id 1       # VLM GPU
```
