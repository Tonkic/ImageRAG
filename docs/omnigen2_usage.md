# OmniGen2 使用指南

> **[❗ 重要说明]**
> OmniGen2 对显存和算力有较高要求。超参数调优和管线迭代的**代码编写**必须在**本地**完成并校验语法，
> **实际推理测试**请同步至**服务器**运行以获得最佳性能。

本文档基于 OmniGen2 官方建议，汇总超参数调优、显存管理、生成质量提升等最佳实践。

---

## 一、核心超参数

### 1. `text_guidance_scale`

Classifier-Free Guidance (CFG) 强度，控制输出图像对文本 prompt 的遵从程度。

- 值越高 → 输出越忠实于文本描述
- 值越低 → 输出更"自由"，多样性更高
- **项目默认值**：`2.5`

### 2. `image_guidance_scale`

控制输出图像对输入参考图像的相似程度。

| 场景 | 推荐范围 | 说明 |
|------|----------|------|
| **图像编辑 (Image Editing)** | 1.2 ~ 2.0 | 保留原图结构的同时允许文本修改 |
| **上下文生成 (In-context Generation)** | 2.5 ~ 3.0 | 保持更多输入图像细节 |

> **权衡**：值越高越忠实于参考图像的结构和风格，但可能忽略部分文本指令；值越低（~1.5）则给予文本 prompt 更大的影响力。

- **项目默认值**：`1.6`

### 3. `max_pixels`

输入图像像素上限（宽 × 高），超过限制时自动按比例缩放。

- **默认值**：`1024 × 1024 = 1,048,576`
- **提示**：遇到 OOM 时可适当降低此值

### 4. `max_input_image_side_length`

输入图像最大边长限制。

### 5. `negative_prompt`

告诉模型不希望在生成图像中看到什么。

```
blurry, low quality, text, watermark, lowres, ugly, deformed
```

> **提示**：不同的 negative prompt 会显著影响生成质量，建议多尝试。不确定时使用上述默认值即可。

### 6. `num_inference_steps`

ODE solver 的离散化步数。

- **默认值**：`50`
- 步数越多质量越高，但推理越慢

### 7. `scheduler`

调度器选择：`euler`（默认）或 `dpmsolver++`。

- `dpmsolver++` 在较少步数下可能表现更好

### 8. `cfg_range_start` / `cfg_range_end`

定义 CFG 生效的时间步范围。减小 `cfg_range_end` 可显著降低推理时间，对质量影响极小。

---

## 二、加速推理

### 1. TaylorSeer（默认开启）

```bash
# 默认已启用，无需额外指定
python OmniGenV2_TAC_DINO_Importance_Aircraft.py --device_id 0 ...

# 若需关闭
python OmniGenV2_TAC_DINO_Importance_Aircraft.py --disable_taylorseer ...
```

- 推理速度最高提升 **2×**
- 对生成质量影响可忽略
- **与 TeaCache 互斥**，同时启用时 TeaCache 会被自动忽略

### 2. TeaCache（默认关闭）

```bash
python OmniGenV2_TAC_DINO_Importance_Aircraft.py --enable_teacache --teacache_thresh 0.05 ...
```

| 参数 | 说明 |
|------|------|
| `--enable_teacache` | 开启 TeaCache |
| `--teacache_thresh` | L1 距离阈值，控制缓存触发频率 |

- 默认阈值 `0.05` 约带来 **30%** 加速
- 提高阈值可进一步降低延迟，但可能损失细节
- **与 TaylorSeer 互斥**

> **注意**：项目中 `--teacache_thresh` 默认设为 `0.4`（面向 fine-grained 场景更激进的阈值），官方默认为 `0.05`。

---

## 三、显存管理

### 1. CPU Offload — `enable_model_cpu_offload`（默认开启）

```bash
# 默认已启用，无需额外指定
python OmniGenV2_TAC_DINO_Importance_Aircraft.py --device_id 0 ...

# 若需关闭（显存充足时可获得更快速度）
python OmniGenV2_TAC_DINO_Importance_Aircraft.py --disable_offload ...
```

- VRAM 占用降低约 **50%**
- 对推理速度影响极小
- 原理：不使用的模型权重自动卸载到 CPU RAM

### 2. Sequential CPU Offload — `enable_sequential_cpu_offload`

- VRAM 占用降至 **< 3GB**
- 显著降低推理速度
- 原理：按子模块逐个加载到 GPU

### 3. VAE 优化

项目中已默认开启：

```python
pipe.vae.enable_tiling()   # 分块处理，降低 VAE 峰值显存
pipe.vae.enable_slicing()  # 切片处理
```

### 4. 双 GPU 策略

将生成模型和 VLM 分配到不同 GPU：

```bash
python OmniGenV2_TAC_DINO_Importance_Aircraft.py \
    --device_id 0 \          # OmniGen2 生成模型
    --vlm_device_id 1        # Qwen3-VL / 检索模型 / DINOv3
```

### 5. 4-bit 量化（Qwen3-VL）

```bash
python OmniGenV2_TAC_DINO_Importance_Aircraft.py --qwen_4bit ...
```

- 使用 bitsandbytes NF4 量化 Qwen3-VL
- 进一步降低 VLM 显存占用
- **默认关闭**

---

## 四、提升生成质量

### 1. 使用高质量输入图像

- 分辨率建议 **> 512 × 512**
- 小图或模糊图像会导致低质量输出

### 2. 详细的文本描述

- 清晰描述 **要改变什么** 以及 **期望的效果**
- **较长的 prompt 通常优于较短的**
- 对场景和主体交互的详细描述会有额外收益
- **优先使用英文** prompt（模型对英文表现最佳）

### 3. 增强主体一致性 (Subject Consistency)

当生成图像与输入参考图像不一致时，尝试以下方法：

| 方法 | 说明 |
|------|------|
| 使用更大尺寸的参考图像 | 主体在画面中占比更大效果更好 |
| 提高 `image_guidance_scale` | 例如设置为 `3.0`（可能导致轻微过曝或油腻感） |
| 使用模板化 prompt | `"she/he ..., maintaining her/his facial features, hairstyle, and other attributes."` |
| 增加每个 prompt 的生成数量 | 多张生成结果中挑选最佳 |

### 4. 多图编辑 (In-context Edit) Prompt 模板

基于多张输入图像的编辑推荐格式：

```
Edit the first image: add/replace (the [object] with) the [object] from the second image.
[detailed description for your target image].
```

**示例**：

```
Edit the first image: add the man from the second image.
The man is talking with a woman in the kitchen.
```

> 目标图像的描述应尽可能详细。

---

## 五、项目默认参数汇总

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable_offload` | **ON** | CPU offloading 降低 VRAM |
| `--enable_taylorseer` | **ON** | TaylorSeer 加速推理 |
| `--enable_teacache` | OFF | TeaCache（与 TaylorSeer 互斥） |
| `--qwen_4bit` | OFF | Qwen3-VL 4-bit 量化 |
| `--text_guidance_scale` | 2.5 | 文本引导强度 |
| `--image_guidance_scale` | 1.6 | 图像引导强度 |
| `--height` / `--width` | 512 | 输出分辨率 |
| `--num_inference_steps` | 50 | 推理步数 |
| `--negative_prompt` | `"blurry, low quality, ..."` | 负面提示 |
| `--seed` | 0 | 随机种子 |
| `--max_retries` | 3 | 最大重试次数（Step 5） |
| `--var_k` | 10 | VAR 候选数量 |
| `--dino_lambda_init` | 0.3 | DINO 结构先验初始权重 |
| `--dino_lambda_step` | 0.15 | 每次重试 λ 增量 |
| `--dino_lambda_max` | 0.8 | λ 上限 |
| `--tac_pass_threshold` | 6.0 | TAC 通过阈值 |
| `--tac_early_stop_threshold` | 8.0 | TAC 早停阈值 |

---

## 六、典型运行命令

```bash
# 基础运行（offload + taylorseer 默认开启）
python OmniGenV2_TAC_DINO_Importance_Aircraft.py \
    --device_id 0 \
    --task_index 0 \
    --total_chunks 1 \
    --omnigen2_path ./OmniGen2 \
    --retrieval_method LongCLIP \
    --use_local_model_weight

# 双 GPU + 4-bit 量化
python OmniGenV2_TAC_DINO_Importance_Aircraft.py \
    --device_id 0 \
    --vlm_device_id 1 \
    --retrieval_method LongCLIP \
    --use_local_model_weight \
    --qwen_4bit

# 关闭 offload（显存充足时更快）
python OmniGenV2_TAC_DINO_Importance_Aircraft.py \
    --device_id 0 \
    --disable_offload \
    --retrieval_method CLIP

# 使用 TeaCache 替代 TaylorSeer
python OmniGenV2_TAC_DINO_Importance_Aircraft.py \
    --device_id 0 \
    --disable_taylorseer \
    --enable_teacache \
    --teacache_thresh 0.05 \
    --retrieval_method LongCLIP
```
