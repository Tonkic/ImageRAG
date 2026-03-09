# 脚本文档

> **[❗ 重要说明]**
> 在修改或编写新脚本时，请务必在**本地环境**进行开发和代码语法检查。
> **在运行任何脚本前，请确保已激活 `t2i` conda 环境 (`conda activate t2i`)**。
>
> **【代码同步流程】**
> 1. 本地开发无误后，**首先通过 Git 推送代码至 GitHub 仓库** (`https://github.com/Tonkic/t2i-research-lab`)：
>    ```bash
>    git add .
>    git commit -m "update message"
>    git push origin main
>    ```
> 2. 接着，再使用本文档底部的“同步脚本”将改动上传至服务器执行。

本文档详细说明项目中所有脚本的功能和使用方法。脚本分为 **核心工具模块**、**实验脚本**、**评估脚本**、**训练脚本** 和 **辅助脚本** 五大类。

---

## 一、核心工具模块

这些模块被实验脚本导入使用，是整个管线的基础设施。

### 1. rag_utils.py

| 项目 | 详情 |
|------|------|
| **用途** | 全局基础设施：种子设定、资源监控、Token 统计、经验库、VLM 封装 |
| **依赖方** | 所有实验脚本 |

**关键组件**：

| 类/函数 | 说明 |
|---------|------|
| `seed_everything(seed)` | 设置全局随机种子 (random, numpy, torch) |
| `ResourceMonitor` | 后台线程监控 CPU/RAM/GPU 使用率，支持保存图表 |
| `RUN_STATS` | 全局运行统计字典 |
| `UsageTrackingClient` | 包装 OpenAI client，自动统计 input/output token 用量 |
| `ExperienceLibrary` | 经验库，分全局规则和类别特定规则两层，支持去重 |
| `LocalQwen3VLWrapper` | 兼容 OpenAI API 接口的本地 Qwen3-VL 推理封装 |

---

### 2. taxonomy_aware_critic.py

| 项目 | 详情 |
|------|------|
| **用途** | **TAC (Taxonomy-Aware Critic)** 核心模块 |
| **依赖方** | 所有 `*_TAC_*.py` 实验脚本 |

**关键函数**：

| 函数 | 说明 |
|------|------|
| `encode_image(path, max_size)` | 图像压缩并 base64 编码 |
| `message_gpt(msg, client, image_paths, model)` | VLM 通信封装（支持重试、图像优先插入） |
| `generate_knowledge_specs(prompt, client, model, domain)` | 根据领域 (aircraft/birds/generic) 生成视觉识别规格 |
| `taxonomy_aware_diagnosis(...)` | **核心评估**：三层评分协议，包含 JSON Token 截断自愈与严格字数截断机制 |
| `input_interpreter(prompt, client, model)` | 输入解析：提取主体、属性、创意细节 |

**评分协议**：

| 评分层级 | 分数 | 含义 |
|---------|------|------|
| Tier C | 0-3 | 完全错误概念（如飞机生成为汽车） |
| Tier B | 4-5 | 子类型错误（如 737-300 生成为 737-800） |
| Tier A | 6.0 | 分类正确（基准分） |
| 细节加分 | +4 | 结构/属性/环境/质量细节 |

### 2.1 新增多轴 Critic 模块

| 模块 | 用途 | 推荐场景 |
|------|------|------|
| `critic_common.py` | Critic 公共封装：图像编码、消息构造、结构化 JSON 调用 | 所有新 Critic 共享底座 |
| `fine_grained_alignment_critic.py` | 细粒度属性对齐评估 | Aircraft/CUB 子类型与关键结构检查 |
| `identity_preservation_critic.py` | 参考图-生成图身份保持评估 | 检索增强生成、DINO 注入实验 |
| `visual_realism_critic.py` | 图像真实感与伪影分析 | 画质筛查、失败案例归因 |
| `overall_t2i_alignment_critic.py` | 整体文本-图像对齐分析 | Prompt 覆盖度分析 |
| `multi_axis_critic.py` | 聚合四类 Critic 输出 | 离线评测、误差拆解、报告生成 |

**当前建议**：

- 若要替换 Step 4 的主 gate，优先尝试 `identity_preservation_critic.py`
- 若要保留现有重生成逻辑，继续使用 `taxonomy_aware_critic.py` 负责 `refined_prompt`
- 若要做 ablation 或结果诊断，补跑 `multi_axis_critic.py`

---

### 3. memory_guided_retrieval.py

| 项目 | 详情 |
|------|------|
| **用途** | **MGR (Memory-Guided Retrieval)** 动态检索模块 |
| **行数** | ~1424 行 |
| **依赖方** | 所有 `*_MGR_*.py` 和 `*_VAR_*.py` 实验脚本 |

**关键组件**：

| 类/函数 | 说明 |
|---------|------|
| `ImageRetriever` | 统一的图像检索器，支持多种检索方法 |
| `HybridRetriever` | BM25 + 向量检索融合 (alpha 加权) |
| `retrieve_img_per_caption(captions, ...)` | 统一入口函数，按 method 路由到对应检索器 |
| `get_clip_similarities(...)` | CLIP ViT-B/32 检索（含缓存、混合模式） |
| `get_longclip_similarities(...)` | Long-CLIP 检索 (248 tokens) |
| `get_siglip_similarities(...)` | SigLIP 检索 |
| `get_siglip2_similarities(...)` | SigLIP2 检索 |

**检索方法路由**：

```python
# --retrieval_method 可选值:
# CLIP, LongCLIP, SigLIP, SigLIP2, Qwen2.5-VL, Qwen3-VL
```

**嵌入缓存机制**：
- 首次检索时自动计算嵌入并存储为 `{method}_embeddings_b{offset}.pt`，后续直接加载。
- **全局参数级缓存 (Global Model Caching)**：相同 `method` (如 LongCLIP) 会跨实例复用 VRAM 中的模型，仅在首次加载时占用显存，极大地节约了多库混合检索（如 Aircraft + CUB + ImageNet）时的开销。

---

---

### 4. custom_pipeline.py

| 项目 | 详情 |
|------|------|
| **用途** | OmniGen2 自定义管线（支持多种融合模式） |
| **依赖方** | AR 相关的流水线实验脚本 |

**关键类**：

| 类 | 说明 |
|------|------|
| `CustomOmniGen2Pipeline` | 基础自定义管线：支持预计算 latent 注入 |
| `CustomOmniGen2DiTLateFusionPipeline` | **晚期融合**：AR 只接收文本，图像通过 VAE latent 直接注入 DiT |
| `CustomOmniGen2AREarlyFusionPipeline` | **早期融合**：图像被 MLLM 转为 token 影响 Hidden States |

---

## 二、实验脚本

### 命名规范

实验脚本遵循统一的命名格式：

```
{Generator}_{Critic}_{Retrieval}_{Dataset}[_Variant].py
```

| 缩写 | 全称 | 说明 |
|------|------|------|
| **TAC** | Taxonomy-Aware Critic | 分层评分系统 |
| **BC** | Binary Critic | 二元评分基线 |
| **SR** | Static Retrieval | 纯 Top-K 静态检索 |
| **MGR** | Memory-Guided Retrieval | 记忆引导动态检索 |
| **VAR** | VLM-As-Reranker | VLM 精细化重排序 |
| **GDA** | Gaussian Discriminant Analysis | 高斯判别分析 |
| **noRAG** | No Retrieval | 仅 prompt 细化/拒绝采样 |
| **LateFusion** | Late Fusion | DiT 晚期融合 |
| **EarlyFusion** | Early Fusion | AR 早期融合 |
| **Latent** | Precomputed Latent | 预计算 latent 模式 |

### 通用参数

所有实验脚本支持以下通用参数：

```bash
# 核心配置
--device_id 0                    # GPU 设备 ID
--vlm_device_id 1                # VLM 专用 GPU（可选）
--task_index 0                   # 分块任务索引（多卡并行用）
--total_chunks 1                 # 总分块数

# 模型路径
--omnigen2_path ./OmniGen2       # OmniGen2 仓库路径
--omnigen2_model_path OmniGen2/OmniGen2  # 模型权重路径
--transformer_lora_path <path>   # LoRA 适配器路径（可选）

# VLM 配置
--use_local_model_weight         # 使用本地模型权重
--local_model_weight_path Qwen3-VL-4B-Instruct  # 本地模型路径
--qwen_4bit                      # 4-bit 量化
--openai_api_key <key>           # API Key（不用本地模型时）
--llm_model Qwen/Qwen3-VL-30B-A3B-Instruct  # API 模型名

# 生成参数
--seed 0
--max_retries 3
--height 512 --width 512
--image_guidance_scale 1.6
--text_guidance_scale 2.5
--use_negative_prompt            # 启用负面提示词注入 (默认: 关闭)
--negative_prompt "blurry, low quality, text, watermark"

# 检索配置
--retrieval_method LongCLIP      # CLIP/LongCLIP/SigLIP/SigLIP2/Qwen3-VL
--embeddings_path datasets/embeddings/imagenet
--retrieval_cpu                  # 检索模型使用 CPU
--use_hybrid_retrieval           # 启用混合检索 (Vector + BM25)
--var_k 10                       # VAR 候选数

# 优化选项
--enable_offload                 # CPU offloading
--enable_taylorseer              # TaylorSeer 加速
```

---

### OmniGenV2 核心流水线 (Aircraft 数据集)

| 脚本 | 架构 | 说明 | 运行命令 |
|------|------|------|----------|
| `OmniGenV2_TAC_DINO_Importance_Aircraft.py` | TAC + DINO + VAR | **5步流水线**：非对称解析→双阶段检索→DINO注入生成→空间诊断→反思重生成 | `python OmniGenV2_TAC_DINO_Importance_Aircraft.py --device_id 0 --retrieval_method LongCLIP --enable_teacache --use_local_model_weight` |
| `OmniGenV2_TAC_DINO_Importance_Aircraft_AR.py` | TAC + DINO + VAR (AR Fusion) | **5步流水线 (Early Fusion)**：使用 AR Early Fusion 引入图像 | `python OmniGenV2_TAC_DINO_Importance_Aircraft_AR.py --device_id 0 --retrieval_method LongCLIP --use_local_model_weight` |

**5 步流水线架构**：

| 步骤 | 名称 | 功能 |
|------|------|------|
| Step 1 | Input Interpreter (非对称解析) | 将 prompt 拆分为 high_importance (纯实体→检索) + low_importance (环境风格→生成) |
| Step 2 | Dual-Stage Retrieval (护城河过滤) | LongCLIP 召回 + VLM-As-Reranker 验证，选出最佳"证件照"级参考图 |
| Step 3 | DINO-Injected Generation (结构先验注入) | DINOv3 提取 2D Patch 特征，调制 image_guidance_scale 注入结构先验 |
| Step 4 | TAC Spatial Diagnosis (空间级诊断) | TAC 三层评分 + DINOv3 结构相似度 + 空间级错误分类 (Global/Local)，并可扩展接入 Identity / Fine-grained / Realism / Overall Alignment Critic |
| Step 5 | Reflexive Re-generation (带外挂的重试) | 负向提示注入 + DINO λ 逐步升级 (结构约束递增) |

**DINO 特有参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dino_lambda_init` | 0.3 | 初始 DINO 结构先验权重 λ |
| `--dino_lambda_step` | 0.15 | 每次重试 λ 递增量 |
| `--dino_lambda_max` | 0.8 | λ 最大阈值 |
| `--tac_pass_threshold` | 6.0 | TAC 通过分数 (Tier A) |
| `--tac_early_stop_threshold` | 8.0 | TAC 早停分数 (优秀) |
| `--use_sam2_matting` | False | 启用 SAM2 进行背景隔离，提取纯净主体防止背景污染 |
| `--decouple_threshold` | 0.25 | DINO 潜空间衰减倒计时，避免颜色/风格崩坏 (0.0=关闭) |

### OmniGenV2 消融实验 (Ablation)

为了证明 5 步流水线各步骤的作用，提供以下消融版本：

| 脚本 | 缺失组件 | 说明 |
|------|----------|------|
| `OmniGenV2_Ablation_noInputDecomp*.py` | 无 Input Interpreter | 直接将完整 prompt 进行检索，无主次划分 |
| `OmniGenV2_Ablation_noVAR*.py` | 无 VLM-as-Reranker | 使用纯视觉检索(如 LongCLIP)的 Top-1，缺乏细粒度验证 |
| `OmniGenV2_Ablation_noDINO*.py` | 无 DINO 特征注入 | 纯注意力机制生成，无显式结构先验指导 |
| `OmniGenV2_Ablation_noTAC*.py` | 无 TAC 分层诊断 | 使用基础 VLM 评分，缺乏细致的局部/全局错误分类及重试 |

*注意：每种消融实验均包含普通版和对应的 `_AR` (Autoregressive Early Fusion) 版本。*

---

## 三、评估脚本

### evaluate_groundtruth.py

| 项目 | 详情 |
|------|------|
| **用途** | GroundTruth 基线评估：用真实测试集图像建立指标上界 |

**评估指标**：
- CLIPScore
- ImageReward
- PickScore
- VQAScore
- BLIPv2Score
- HPSv2Score / HPSv2.1Score
- MPS
- FGA_BLIP2Score

```bash
python evaluate_groundtruth.py
# 输出: groundtruth_metrics.txt
```

---

### evaluate_all_recursive.py

| 项目 | 详情 |
|------|------|
| **用途** | 递归扫描 `results/` 目录并评估所有实验 |
| **行数** | ~789 行 |

**评估指标**：

| 指标 | 说明 |
|------|------|
| CLIP Score | 图文一致性 (OpenCLIP ViT-B-32) |
| SigLIP Score | SigLIP 图文一致性 |
| DINOv3 Score | 图像保真度 (ViT-B/16) |
| KID | Kernel Inception Distance |
| FID | Fréchet Inception Distance |
| IS | Inception Score |
| Laplacian Variance | 图像清晰度 |
| CCMD | Class-Conditional Mahalanobis Distance |
| SVCG | Semantic-Visual Consistency Gap |

```bash
python evaluate_all_recursive.py
# 输出: 每个实验目录下 logs/evaluation_metrics.txt + 控制台汇总表
```

---

### evaluate_evalscope.py

| 项目 | 详情 |
|------|------|
| **用途** | 使用 EvalScope 框架进行标准 T2I 指标评估（内存优化版） |
| **行数** | ~504 行 |
| **特点** | 逐指标加载/释放模型，减少 VRAM 峰值 |

```bash
python evaluate_evalscope.py --result_dir results/LongCLIP/2025.3.1/experiment_name
```

---

### export_results_table.py

| 项目 | 详情 |
|------|------|
| **用途** | 汇总所有实验结果为表格 |

```bash
python export_results_table.py
# 输出: consolidated_results.csv, consolidated_results.md
```

---

## 四、训练脚本

### train_vlm2vec.py

| 项目 | 详情 |
|------|------|
| **用途** | 训练 VLM2Vec：Qwen3-VL + LoRA 图文检索微调 |
| **特点** | Family-level Hard Negative Mining (HierarchicalBatchSampler) |
| **数据集** | FGVC-Aircraft (variant + family 标签) |

```bash
python train_vlm2vec.py
```

---

### train_longclip_to_vae.py

| 项目 | 详情 |
|------|------|
| **用途** | 训练 Projector：1D LongCLIP embedding → 3D VAE Latent |
| **架构** | MLP → Reshape → ConvTranspose2d 上采样 |

```bash
python train_longclip_to_vae.py
```

---

## 五、辅助脚本

### 数据预处理

| 脚本 | 说明 | 运行命令 |
|------|------|----------|
| `precompute_latents.py` | 预计算 OmniGen2 VAE latent | `python precompute_latents.py --omnigen2_path ./OmniGen2` |
| `preprocess_background_removal.py` | BiRefNet 去背景预处理 | `python preprocess_background_removal.py` |
| `read_imagenet_meta.py` | 读取 ImageNet 元数据 | `python read_imagenet_meta.py` |

### 工具脚本

| 脚本 | 说明 | 运行命令 |
|------|------|----------|
| `list_files.py` | 列出项目文件 | `python list_files.py` |
| `organize_results_by_timestamp.py` | 按时间戳整理结果 | `python organize_results_by_timestamp.py` |
| `check_evalscope.py` | 检查 EvalScope 配置 | `python check_evalscope.py` |
| `check_qwen3_config.py` | 检查 Qwen3-VL 配置 | `python check_qwen3_config.py` |

### 演示脚本

| 脚本 | 说明 | 运行命令 |
|------|------|----------|
| `ZImageDemo.py` | Z-Image-Turbo 推理演示 | `python ZImageDemo.py` |

### 同步脚本

| 脚本 | 说明 |
|------|------|
| `sync_to_server.sh` / `sync_to_server.ps1` | 同步代码到服务器 |
| `download_results.sh` / `download_results.ps1` | 从服务器下载结果 |
| `pull_from_server.ps1` | 从服务器拉取文件 |
| `hfd.sh` | HuggingFace 模型下载辅助脚本 |

---

## 六、多卡并行运行

实验脚本支持通过 `--task_index` 和 `--total_chunks` 实现多卡并行：

```bash
# 4 卡并行（每卡处理 1/4 的类别）
python OmniGenV2_TAC_SR_Aircraft.py --device_id 0 --task_index 0 --total_chunks 4 &
python OmniGenV2_TAC_SR_Aircraft.py --device_id 1 --task_index 1 --total_chunks 4 &
python OmniGenV2_TAC_SR_Aircraft.py --device_id 2 --task_index 2 --total_chunks 4 &
python OmniGenV2_TAC_SR_Aircraft.py --device_id 3 --task_index 3 --total_chunks 4 &
```

---

## 七、结果输出结构

所有实验结果保存在 `results/` 目录下，按以下层级组织：

```
results/
└── {retrieval_method}/          # e.g., LongCLIP/
    └── {date}/                  # e.g., 2025.3.1/
        └── {experiment_name}_{time}/  # e.g., OmniGenV2_TAC_SR_Aircraft_14-30-00/
            ├── images/          # 生成的图像
            │   ├── {class_name}/
            │   │   ├── final.png
            │   │   └── retry_{n}.png
            │   └── ...
            ├── logs/            # 运行日志
            │   ├── run_log.json
            │   └── evaluation_metrics.txt
            └── latent_experience_bank.json  # 经验库（如适用）
```
