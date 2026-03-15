# ImageRAG 项目文档

> **[❗ 重要说明]**
> 本项目采用 **“本地开发 + 服务器运行”** 范式。
> 所有代码逻辑修改、新增功能及**基础语法检查**均应在**本地开发环境**完成。
> **在运行任何脚本前，请确保已激活 `t2i` conda 环境 (`conda activate t2i`)**。
>
> **【代码同步流程】**
> 1. 本地开发并测试通过后，请**先将代码提交并 Push 到 GitHub 仓库**：
>    ```bash
>    git add .
>    git commit -m "update message"
>    git push origin main
>    ```
>    (仓库地址: `https://github.com/Tonkic/t2i-research-lab`)
> 2. 之后再通过同步脚本 (如 `sync_to_server.ps1`) 上传至远端服务器进行模型运算。

## 项目概述

**ImageRAG** 是一个基于检索增强生成 (RAG) 的细粒度文本到图像生成框架。

**论文题目**: *Taxonomy-Aware Criticism and Memory-Guided Retrieval for Robust Text-to-Image Generation* (TAC-MGR)

**原始基础**: 基于 [ImageRAG (arXiv:2502.09411)](https://arxiv.org/abs/2502.09411) —— *Dynamic Image Retrieval for Reference-Guided Image Generation* by Shalev-Arkushin et al., 2025.

**核心思想**: T2I 模型在细粒度生成中常常失败（如特定飞机型号、鸟类品种），已有 RAG 方法缺乏动态纠错能力。TAC-MGR 整合了：
- **Taxonomy-Aware Criticism (TAC)** —— 基于分类学的分层评估系统，超越简单的二元判断
- **Memory-Guided Retrieval (MGR)** —— 记忆引导检索，动态更新全局记忆来优化未来的图像检索

## 文档目录

| 文档 | 说明 |
|------|------|
| [datasets.md](datasets.md) | 数据集信息、路径和使用方法 |
| [models.md](models.md) | 模型信息、路径和使用方法 |
| [scripts.md](scripts.md) | 各脚本的作用和使用方法 |

## Critic 模块更新

`src/critical` 现已统一为 `*_critic.py` 命名，并新增一组可组合的多轴 Critic：

| 模块 | 作用 |
|------|------|
| `taxonomy_aware_critic.py` | 现有 TAC 主模块，负责 taxonomy gate、结构化诊断、`refined_prompt` 与 `retrieval_queries` 生成 |
| `fine_grained_alignment_critic.py` | 单图细粒度属性一致性检查，适合 subtype / 关键部件校验 |
| `identity_preservation_critic.py` | 双图身份保持评估，比较参考图与生成图是否仍是同一细粒度实体 |
| `visual_realism_critic.py` | 图像物理合理性与伪影检查 |
| `overall_t2i_alignment_critic.py` | Prompt 与图像整体语义覆盖度评估 |
| `multi_axis_critic.py` | 统一聚合入口，可一次返回四个维度的结果 |
| `critic_common.py` | 公共消息构造、图像编码与 JSON 解析调用封装 |

当前建议：

- 若实验目标是 **Aircraft / CUB 细粒度保真**，优先关注 `identity_preservation_critic.py` 与 `fine_grained_alignment_critic.py`
- 若实验目标是 **重生成修复链路**，继续保留 `taxonomy_aware_critic.py` 负责 `refined_prompt` / `retrieval_queries`
- 若需要分析式评估报告，可额外并行调用 `multi_axis_critic.py`

## 快速开始

### 环境安装

```bash
conda env create -f environment.yml
conda activate t2i
```

### 核心依赖

- Python 3.10
- PyTorch 2.4.1+
- transformers, diffusers, open-clip-torch
- openai, peft, rank-bm25, timm

### 典型运行命令

```bash
# TAC + DINO + VAR 5步精细生成流水线 (Aircraft)
python OmniGenV2_TAC_DINO_Importance_Aircraft.py \
    --device_id 0 \
    --task_index 0 \
    --total_chunks 1 \
    --omnigen2_path ./OmniGen2 \
    --retrieval_method LongCLIP \
    --use_local_model_weight

# IPC-AR 主实验（推荐论文主线配置）
python src/experiments/OmniGenV2_IPC_AR.py \
    --device_id 0 \
    --task_index 0 \
    --total_chunks 1 \
    --omnigen2_path ./OmniGen2 \
    --retrieval_method LongCLIP \
    --retrieval_datasets aircraft cub imagenet \
    --text_api_key YOUR_API_KEY \
    --vl_api_key YOUR_API_KEY \
    --decouple_threshold 0.65 \
    --max_retries 2

# IPC-AR 换用 SigLIP2 检索（自动对齐模型，无需额外参数）
python src/experiments/OmniGenV2_IPC_AR.py \
    --device_id 0 \
    --task_index 0 \
    --total_chunks 1 \
    --omnigen2_path ./OmniGen2 \
    --retrieval_method SigLIP2 \
    --retrieval_datasets aircraft cub imagenet \
    --text_api_key YOUR_API_KEY \
    --vl_api_key YOUR_API_KEY \
    --decouple_threshold 0.65 \
    --max_retries 2

# IPC-AR 主表消融：w/o Stage-2 Rerank
python src/experiments/OmniGenV2_IPC_AR_Ablation_noStage2Rerank.py \
    --device_id 0 \
    --task_index 0 \
    --total_chunks 1 \
    --omnigen2_path ./OmniGen2 \
    --retrieval_method LongCLIP \
    --retrieval_datasets aircraft cub imagenet \
    --text_api_key YOUR_API_KEY \
    --vl_api_key YOUR_API_KEY \
    --decouple_threshold 0.65 \
    --max_retries 2
```

## 整体管线流程

```
用户 Prompt
    ↓
[Step 1 Input Interpreter] → 拆分 High Importance(实体) 与 Low Importance(环境/风格)
    ↓
[Step 2 Dual-Stage Retrieval] → LongCLIP 初筛 + VLM-As-Reranker 精排参考图
    ↓
[Step 3 DINO Injected Generation] → OmniGen2 生成图像，动态注入 DINOv3 视觉结构先验
    ↓
[Step 4 TAC Spatial Diagnosis] → TAC 三层评分(0-10) + DINOv3 结构相似度 + 空间局部错误点定位
    ├─ 可扩展 Fine-grained Alignment Critic → 检查发动机/尾翼/翼尖/机身等属性
    ├─ 可扩展 Identity Preservation Critic → 比较参考图与生成图是否仍为同一子型号
    ├─ 可扩展 Visual Realism Critic → 检查伪影、结构扭曲与材质异常
    └─ 可扩展 Overall T2I Alignment Critic → 检查 Prompt 覆盖与语义偏移
    ↓ (score < early_stop_threshold?)
[Step 5 Reflexive Re-Generation] → 调高 DINO λ 特征权重，按诊断结果补充负向约束重试
    ↓
[Loop up to max_retries] → 迭代直到满意、达到早停或最大重试次数
    ↓
[Save FINAL image + logs] → 保存最终的高保真图像、中间尝试版本以及诊断日志
```

## 项目目录结构

```
imageRAG/
├── docs/                          # 项目文档
├── datasets/                      # 数据集
│   ├── fgvc-aircraft-2013b/       # FGVC-Aircraft 数据集
│   ├── ILSVRC2012_train/          # ImageNet 训练集
│   └── embeddings/                # 预计算的向量嵌入
├── OmniGen2/                      # OmniGen V2 生成模型
├── CLIP/                          # OpenAI CLIP 模型
├── Long-CLIP/                     # Long-CLIP 扩展模型
├── Qwen3-VL-4B-Instruct/         # Qwen3-VL 视觉语言模型
├── Z-Image-Turbo/                 # Z-Image-Turbo 生成模型 (Demo)
├── dinov3/                        # DINOv3 视觉特征模型
├── sam2/                          # SAM2 分割模型
├── FlagEmbedding/                 # FlagEmbedding 检索框架
├── results/                       # 实验结果输出
├── src/
│   ├── experiments/
│   │   └── OmniGenV2_*.py         # OmniGen2 核心融合与消融实验脚本
│   ├── critical/
│   │   ├── taxonomy_aware_critic.py           # TAC 主诊断与修复建议生成
│   │   ├── fine_grained_alignment_critic.py   # 细粒度属性一致性评估
│   │   ├── identity_preservation_critic.py    # 参考图-生成图身份保持评估
│   │   ├── visual_realism_critic.py           # 真实感与伪影评估
│   │   ├── overall_t2i_alignment_critic.py    # 整体文本-图像对齐评估
│   │   ├── multi_axis_critic.py               # 多轴统一入口
│   │   └── critic_common.py                   # Critic 公共工具
│   ├── retrieval/
│   │   └── memory_guided_retrieval.py # (MGR) 混合检索逻辑（含 SigLIP2 自动对齐）
│   └── utils/
│       ├── custom_pipeline.py         # 自定义管线 (DualPath / LateFusion / EarlyFusion)
│       ├── rag_utils.py               # 核心工具库
│       └── utils.py                   # 通用辅助工具
├── evaluate_*.py                  # 评估脚本 (EvalScope集成)
├── train_*.py                     # 训练脚本
└── environment.yml                # Conda 环境配置
```
