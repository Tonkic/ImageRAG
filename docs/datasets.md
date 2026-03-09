# 数据集文档

> **[❗ 重要说明]**
> 由于数据集规模庞大(如 ImageNet)及检索缓存占用的空间，
> 数据集的存储、嵌入计算和检索均建议在**服务器端**运行。**本地仅用于代码开发和语法验证**。

本项目使用了 4 个主要数据集用于细粒度图像生成与检索实验，同时维护预计算的嵌入（embeddings）缓存以加速检索。

---

## 1. ImageNet (ILSVRC2012)

### 基本信息

| 项目 | 详情 |
|------|------|
| **全称** | ImageNet Large Scale Visual Recognition Challenge 2012 |
| **用途** | 通用图像检索数据库（作为大规模候选图像池） |
| **类别数** | 1,000 类 |
| **图像规模** | ~1.28M 训练图像 |
| **图像格式** | JPEG |
| **分辨率** | 不固定（原始尺寸） |

### 文件路径

| 项目 | 路径 |
|------|------|
| **训练集图像** | `/home/tingyu/imageRAG/datasets/ILSVRC2012_train/` |
| **验证集图像** | `/home/tingyu/imageRAG/datasets/ILSVRC2012_val/` |
| **训练集列表** | `/home/tingyu/imageRAG/datasets/imagenet_train_list.txt` |
| **验证集列表** | `/home/tingyu/imageRAG/datasets/imagenet_val_list.txt` |
| **类别名称** | `/home/tingyu/imageRAG/datasets/imagenet_classes.txt` |
| **嵌入缓存** | `/home/tingyu/imageRAG/datasets/embeddings/imagenet/` |

### 目录结构

```
datasets/
├── ILSVRC2012_train/
│   ├── n01440764/         # 各 synset ID 子目录
│   │   ├── n01440764_18.JPEG
│   │   └── ...
│   ├── n01443537/
│   └── ...
├── imagenet_train_list.txt   # 格式: image_path label
├── imagenet_val_list.txt
└── imagenet_classes.txt      # 类别名称映射
```

### 嵌入缓存

位于 `datasets/embeddings/imagenet/`，支持多种检索模型的预计算嵌入：

| 嵌入类型 | 文件名模式 | 模型 |
|---------|----------|------|
| CLIP | `clip_embeddings_b{offset}.pt` | OpenAI CLIP ViT-B/32 |
| LongCLIP | `longclip_embeddings_b{offset}.pt` | Long-CLIP ViT-L/14 (248 tokens) |
| SigLIP | `siglip_embeddings_b{offset}.pt` | SigLIP ViT-SO400M-14-SigLIP |
| SigLIP2 | `siglip2_embeddings_b{offset}.pt` | SigLIP2 base-patch16-224 |
| Qwen3-VL | `qwen3_vl_embeddings_b{offset}.pt` | Qwen3-VL-4B 视觉编码器 |
| Qwen2.5-VL | `qwen_embeddings_b{offset}.pt` | Qwen2.5-VL |

### 使用方法

在实验脚本中通过 `--retrieval_datasets` 参数指定包含 `imagenet`：

```bash
python OmniGenV2_TAC_DINO_Importance_Aircraft_AR.py \
    --device_id 0 \
    --retrieval_datasets aircraft imagenet \
    --retrieval_method LongCLIP \
    --embeddings_path datasets/embeddings/imagenet
```

在代码中通过 `load_db()` 函数加载：

```python
# memory_guided_retrieval.py / static_retrieval.py
db = load_db()
# db 包含 {"image_paths": [...], "embeddings": tensor}
```

---

## 2. FGVC-Aircraft

### 基本信息

| 项目 | 详情 |
|------|------|
| **全称** | Fine-Grained Visual Classification of Aircraft |
| **用途** | 细粒度飞机型号分类和生成评估 |
| **层级** | 3 层分类法: Family → Manufacturer → Variant |
| **Variant 类别数** | 100 种 |
| **图像规模** | ~10,000 张 |
| **图像格式** | JPEG |
| **数据来源** | The Aircraft Dataset (2013) |

### 文件路径

| 项目 | 路径 |
|------|------|
| **根目录** | `/home/tingyu/imageRAG/datasets/fgvc-aircraft-2013b/` |
| **图像目录** | `/home/tingyu/imageRAG/datasets/fgvc-aircraft-2013b/data/images/` |
| **嵌入缓存** | `/home/tingyu/imageRAG/datasets/embeddings/aircraft/` |
| **预计算 Latent** | `/home/tingyu/imageRAG/datasets/latents/aircraft/` |

### 目录结构

```
datasets/fgvc-aircraft-2013b/
├── README.md
├── data/
│   ├── images/                        # 原始图像
│   │   ├── 0034309.jpg
│   │   └── ...
│   ├── variants.txt                   # 100 个 Variant 类名列表
│   ├── families.txt                   # Family 类名列表
│   ├── manufacturers.txt              # Manufacturer 类名列表
│   ├── images_variant_train.txt       # 训练集 (Variant 级)
│   ├── images_variant_val.txt         # 验证集
│   ├── images_variant_test.txt        # 测试集
│   ├── images_variant_trainval.txt    # 训练+验证集
│   ├── images_family_train.txt        # 训练集 (Family 级)
│   ├── images_family_val.txt
│   ├── images_family_test.txt
│   ├── images_family_trainval.txt
│   ├── images_manufacturer_train.txt  # 训练集 (Manufacturer 级)
│   ├── images_manufacturer_val.txt
│   ├── images_manufacturer_test.txt
│   ├── images_manufacturer_trainval.txt
│   ├── images_train.txt               # 通用训练集
│   ├── images_val.txt                 # 通用验证集
│   ├── images_test.txt                # 通用测试集
│   └── images_box.txt                 # Bounding box 标注
└── evaluation.m                       # MATLAB 评估脚本（原始）
```

### 标注文件格式

**variants.txt** — 每行一个类别名：
```
707-320
727-200
737-200
737-300
...
```

**images_variant_test.txt** — 每行 `图片ID 类别名`：
```
0034309 707-320
0034958 707-320
...
```

### 嵌入缓存

位于 `datasets/embeddings/aircraft/`：

| 嵌入类型 | 文件名模式 |
|---------|----------|
| CLIP | `clip_embeddings_b{offset}.pt` |
| LongCLIP | `longclip_embeddings_b{offset}.pt` |
| SigLIP | `siglip_embeddings_b{offset}.pt` |
| SigLIP2 | `siglip2_embeddings_b{offset}.pt` |
| Qwen3-VL | `qwen3_vl_embeddings_b{offset}.pt` |

### 使用方法

多数实验脚本默认使用 Aircraft 数据集：

```bash
# 标准运行 (Aircraft)
python OmniGenV2_TAC_DINO_Importance_Aircraft.py \
    --device_id 0 \
    --retrieval_method LongCLIP
```

在代码中通过 `DATASET_CONFIG` 配置：

```python
DATASET_CONFIG = {
    "classes_txt": "datasets/fgvc-aircraft-2013b/data/variants.txt",
    "train_list": "datasets/fgvc-aircraft-2013b/data/images_variant_trainval.txt",
    "image_root": "datasets/fgvc-aircraft-2013b/data/images",
    "output_path": f"results/{_rm}/{timestamp}/..."
}
```

---

## 3. 嵌入缓存系统

### 缓存机制

本项目使用预计算嵌入来加速检索过程。嵌入以分片 (batch) 形式存储为 `.pt` 文件。

### 整体结构

```
datasets/embeddings/
├── aircraft/              # Aircraft 数据集嵌入
│   ├── clip_embeddings_b0.pt
│   ├── longclip_embeddings_b0.pt
│   ├── siglip_embeddings_b0.pt
│   ├── siglip2_embeddings_b0.pt
│   └── qwen3_vl_embeddings_b0.pt
└── imagenet/              # ImageNet 数据集嵌入 (数量最多)
    ├── clip_embeddings_b*.pt
    ├── longclip_embeddings_b*.pt
    ├── siglip_embeddings_b*.pt
    ├── siglip2_embeddings_b*.pt
    ├── qwen3_vl_embeddings_b*.pt
    └── qwen_embeddings_b*.pt
```

### 嵌入文件格式

每个 `.pt` 文件是一个 PyTorch dict：

```python
data = torch.load("clip_embeddings_b0.pt")
# data = {
#     "embeddings": tensor,       # [N, D] 特征向量
#     "image_paths": list,        # [N] 对应图像路径
# }
```

### 生成嵌入

嵌入在首次使用某检索方法时自动计算并缓存。也可通过在代码中显式设置来预计算：

```python
from memory_guided_retrieval import ImageRetriever
retriever = ImageRetriever(method="LongCLIP", embeddings_path="datasets/embeddings/aircraft")
# 首次加载时会自动计算并存储嵌入
```

---

## 4. 辅助数据文件

| 文件 | 路径 | 说明 |
|------|------|------|
| **属性文件** | `datasets/attributes.txt` | 额外属性标注 |
| **synset 提取** | `datasets/extract_synsets.py` | ImageNet synset 元数据提取工具 |
