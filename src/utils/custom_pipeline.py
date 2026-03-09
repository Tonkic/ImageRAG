
import os
import sys
import torch
import PIL.Image
from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np

# Adjust path to find OmniGen2
script_dir = os.path.dirname(os.path.abspath(__file__))
# Jump back two levels from src/utils to root
sys.path.append(os.path.abspath(os.path.join(script_dir, "../../OmniGen2")))

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline, FMPipelineOutput
from omnigen2.pipelines.omnigen2.pipeline_omnigen2_chat import OmniGen2ChatPipeline
from omnigen2.models.transformers.repo import OmniGen2RotaryPosEmbed
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
import torch.nn.functional as F


class CustomOmniGen2Pipeline(OmniGen2Pipeline):
    """
    Subclass of OmniGen2Pipeline that supports skipping VAE encoding
    by accepting pre-computed latents.
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.LongTensor] = None,
        negative_prompt_attention_mask: Optional[torch.LongTensor] = None,
        max_sequence_length: Optional[int] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
        input_images: Optional[List[PIL.Image.Image]] = None,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_pixels: int = 1024 * 1024,
        max_input_image_side_length: int = 1024,
        align_res: bool = True,
        num_inference_steps: int = 28,
        text_guidance_scale: float = 4.0,
        image_guidance_scale: float = 1.0,
        cfg_range: Tuple[float, float] = (0.0, 1.0),
        attention_kwargs: Optional[Dict[str, Any]] = None,
        timesteps: List[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        verbose: bool = False,
        step_func=None,
        # [NEW ARGUMENT]
        input_image_latents: Optional[List[List[torch.FloatTensor]]] = None,
        decouple_threshold: float = 0.0,
    ):
        # [OPTIM] Clear Cache
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._text_guidance_scale = text_guidance_scale
        self._image_guidance_scale = image_guidance_scale
        self._cfg_range = cfg_range
        self._attention_kwargs = attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            self.text_guidance_scale > 1.0,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
        )

        dtype = self.vae.dtype
        # 3. Prepare control image
        # [OPTIM] Visual Latent Cache Override
        if input_image_latents is not None:
             # If provided, use these latents directly.
             ref_latents = input_image_latents
        else:
            ref_latents = self.prepare_image(
                images=input_images,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                max_pixels=max_pixels,
                max_side_length=max_input_image_side_length,
                device=device,
                dtype=dtype,
            )

        if input_images is None:
            input_images = []

        if len(input_images) == 1 and align_res:
            # Need tensor shape from ref_latents to determine size
            # If using cache, ref_latents is List[List[Tensor]]
            # Tensor shape: [C, H, W] inside list
            # Usually prepare_image returns List[List[Tensor]]

            # Logic for resizing output to match input
            try:
                # Assuming batch_size=1, first image
                lat_tensor = ref_latents[0][0] # Get first latent
                width, height = lat_tensor.shape[-1] * self.vae_scale_factor, lat_tensor.shape[-2] * self.vae_scale_factor
                ori_width, ori_height = width, height
            except:
                # Fallback if structure mismatches or empty
                ori_width, ori_height = width, height

        else:
            ori_width, ori_height = width, height

            cur_pixels = height * width
            ratio = (max_pixels / cur_pixels) ** 0.5
            ratio = min(ratio, 1.0)

            height, width = int(height * ratio) // 16 * 16, int(width * ratio) // 16 * 16

        if len(input_images) == 0 and input_image_latents is None:
            self._image_guidance_scale = 1

        # 4. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
            self.transformer.config.axes_dim_rope,
            self.transformer.config.axes_lens,
            theta=10000,
        )

        image = self.processing(
            latents=latents,
            ref_latents=ref_latents,
            prompt_embeds=prompt_embeds,
            freqs_cis=freqs_cis,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            device=device,
            dtype=dtype,
            verbose=verbose,
            step_func=step_func,
            decouple_threshold=decouple_threshold,
        )

        # Parent class returns List[Tensor], process each image
        postprocessed_images = []
        for img_tensor in image:
            # Resize to original dimensions
            img_resized = F.interpolate(img_tensor.unsqueeze(0), size=(ori_height, ori_width), mode='bilinear')
            postprocessed_images.append(img_resized)

        # Stack into batch [B, C, H, W] for postprocessing
        if postprocessed_images:
            image_batch = torch.cat(postprocessed_images, dim=0)
            image = self.image_processor.postprocess(image_batch, output_type=output_type)
        else:
            image = []

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return image
        else:
            return FMPipelineOutput(images=image)

    def processing(
        self,
        latents,
        ref_latents,
        prompt_embeds,
        freqs_cis,
        negative_prompt_embeds,
        prompt_attention_mask,
        negative_prompt_attention_mask,
        num_inference_steps,
        timesteps,
        device,
        dtype,
        verbose,
        step_func=None,
        decouple_threshold=0.0
    ):
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
        batch_size = latents.shape[0]

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            num_tokens=latents.shape[-2] * latents.shape[-1]
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        enable_taylorseer = getattr(self, "enable_taylorseer", False)
        if enable_taylorseer:
            from omnigen2.cache_functions import cache_init
            model_pred_cache_dic, model_pred_current = cache_init(self, num_inference_steps)
            model_pred_ref_cache_dic, model_pred_ref_current = cache_init(self, num_inference_steps)
            model_pred_uncond_cache_dic, model_pred_uncond_current = cache_init(self, num_inference_steps)
            self.transformer.enable_taylorseer = True
        elif hasattr(self.transformer, "enable_teacache") and self.transformer.enable_teacache:
            from omnigen2.utils.teacache_util import TeaCacheParams
            teacache_params = TeaCacheParams()
            teacache_params_uncond = TeaCacheParams()
            teacache_params_ref = TeaCacheParams()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # ==========================================================
                # [NEW] Timestep Decoupling Logic (过河拆桥)
                # ==========================================================
                current_ratio = i / len(timesteps)
                if decouple_threshold > 0 and current_ratio >= decouple_threshold:
                    ref_latents = None
                    self._image_guidance_scale = 1.0

                if enable_taylorseer:
                    self.transformer.cache_dic = model_pred_cache_dic
                    self.transformer.current = model_pred_current
                elif hasattr(self.transformer, "enable_teacache") and self.transformer.enable_teacache:
                    teacache_params.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                    self.transformer.teacache_params = teacache_params

                model_pred = self.predict(
                    t=t,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    freqs_cis=freqs_cis,
                    prompt_attention_mask=prompt_attention_mask,
                    ref_image_hidden_states=ref_latents,
                )
                text_guidance_scale = self.text_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1] else 1.0
                image_guidance_scale = self.image_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1] else 1.0

                if text_guidance_scale > 1.0 and image_guidance_scale > 1.0:
                    if enable_taylorseer:
                        self.transformer.cache_dic = model_pred_ref_cache_dic
                        self.transformer.current = model_pred_ref_current
                    elif hasattr(self.transformer, "enable_teacache") and self.transformer.enable_teacache:
                        teacache_params_ref.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_ref

                    model_pred_ref = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=ref_latents,
                    )

                    if enable_taylorseer:
                        self.transformer.cache_dic = model_pred_uncond_cache_dic
                        self.transformer.current = model_pred_uncond_current
                    elif hasattr(self.transformer, "enable_teacache") and self.transformer.enable_teacache:
                        teacache_params_uncond.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_uncond

                    model_pred_uncond = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=None,
                    )

                    model_pred = model_pred_uncond + image_guidance_scale * (model_pred_ref - model_pred_uncond) + \
                        text_guidance_scale * (model_pred - model_pred_ref)
                elif text_guidance_scale > 1.0:
                    if enable_taylorseer:
                        self.transformer.cache_dic = model_pred_uncond_cache_dic
                        self.transformer.current = model_pred_uncond_current
                    elif hasattr(self.transformer, "enable_teacache") and self.transformer.enable_teacache:
                        teacache_params_uncond.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_uncond

                    model_pred_uncond = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=None,
                    )
                    model_pred = model_pred_uncond + text_guidance_scale * (model_pred - model_pred_uncond)

                latents = self.scheduler.step(model_pred, t, latents, return_dict=False)[0]

                latents = latents.to(dtype=dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if step_func is not None:
                    step_func(i, self._num_timesteps)

        if enable_taylorseer:
            del model_pred_cache_dic, model_pred_ref_cache_dic, model_pred_uncond_cache_dic
            del model_pred_current, model_pred_ref_current, model_pred_uncond_current

        latents = latents.to(dtype=dtype)
        if self.vae.config.scaling_factor is not None:
            latents = latents / self.vae.config.scaling_factor
        if self.vae.config.shift_factor is not None:
            latents = latents + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]

        return image



class CustomOmniGen2DiTLateFusionPipeline(CustomOmniGen2Pipeline):
    """
    真正的纯晚期融合：
    1. AR 模型只接收纯文本（无图像 token），输出 Hidden States。
    2. 图像单独通过 VAE 提取 Latents，直接注入给 DiT。
    """
    @torch.no_grad()
    def __call__(self, prompt: str, input_images: List[PIL.Image.Image] = None, **kwargs):

        # 步骤 1：手动提取 VAE Latents（走路径二）
        device = self._execution_device
        dtype = self.vae.dtype

        ref_latents = None
        if input_images is not None:
            # 提前使用 VAE 处理图像
            ref_latents = self.prepare_image(
                images=input_images,
                batch_size=kwargs.get('batch_size', 1),
                num_images_per_prompt=kwargs.get('num_images_per_prompt', 1),
                max_pixels=kwargs.get('max_pixels', 1024 * 1024),
                max_side_length=kwargs.get('max_input_image_side_length', 1024),
                device=device,
                dtype=dtype,
            )

        # 步骤 2：调用修改后的父类
        # 注意：这里我们故意将 input_images 置为 None，这样 AR 端就拿不到图像，彻底跳过 ViT
        # 同时传入 input_image_latents，让 DiT 能拿到视觉条件
        return super().__call__(
            prompt=prompt,
            input_images=None,
            input_image_latents=ref_latents,
            **kwargs
        )


class CustomOmniGen2AREarlyFusionPipeline(OmniGen2ChatPipeline):
    """
    真正的纯早期融合：
    图像被 MLLM 转换为 Token 序列影响 Hidden States。
    我们向 DiT 隐瞒参考图像，强制 DiT 仅依赖 AR 的 Hidden States 生成。
    """
    @torch.no_grad()
    def __call__(self, prompt: str, input_images: List[PIL.Image.Image] = None, **kwargs):

        chat_prompt = self._apply_chat_template(prompt, input_images)
        chat_prompt += "<|img|>"

        # 步骤 1：先让 AR 模型处理图文混合序列，获取输出的 Hidden States (prompt_embeds)
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=chat_prompt,
            do_classifier_free_guidance=kwargs.get('text_guidance_scale', 4.0) > 1.0,
            input_images=input_images, # 图像在这里合法地喂给了 AR (ViT)
            device=self._execution_device
        )

        # 步骤 2：调用底层的生成管线，但切断图像输入
        # 传入已经编码好的 prompt_embeds，并将 input_images 设为 None，这样 VAE 就不会起作用
        images = self.generate_image(
            prompt=None, # 已有 embeds，置空
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            input_images=None, # <--- 核心切断点：DiT 将得不到 VAE 的 ref_latents
            return_dict=False,
            **kwargs
        )

        return FMPipelineOutput(images=images)


class CustomOmniGen2DualPathPipeline(OmniGen2ChatPipeline):
    """
    双重融合路径 (Dual Path)：
    继承官方 OmniGen2ChatPipeline 并覆盖 generate_image 和 __call__，
    允许同时传入图片用于 AR 早期融合，以及 input_image_latents 用于 DiT 晚期融合，
    并完美支持透传额外 kwargs (如 decouple_threshold)。
    """

    # [BUG REPAIR] Borrow the patched processing method from CustomOmniGen2Pipeline to support decouple_threshold
    processing = CustomOmniGen2Pipeline.processing

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        input_images: Optional[List[PIL.Image.Image]] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Bypasses native Chat pipeline's text generation (which discards kwargs),
        formats prompt, and directly passes kwargs down to our overridden generate_image.
        """
        assert isinstance(prompt, str), "prompt must be a string since chat mode only support one prompt per turn"

        chat_prompt = prompt
        # Fallback manual template if _apply_chat_template doesn't do it
        if input_images is not None and len(input_images) == 1 and "<|image_1|>" not in chat_prompt:
             chat_prompt = f"{chat_prompt}. Use reference image <|image_1|>."

        chat_prompt = self._apply_chat_template(chat_prompt, input_images)
        if "<|img|>" not in chat_prompt:
             chat_prompt += "<|img|>"

        return self.generate_image(
            prompt=chat_prompt,
            input_images=input_images,
            return_dict=return_dict,
            **kwargs
        )

    @torch.no_grad()
    def generate_image(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.LongTensor] = None,
        negative_prompt_attention_mask: Optional[torch.LongTensor] = None,
        use_text_encoder_penultimate_layer_feats: bool = False,
        max_sequence_length: Optional[int] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
        input_images: Optional[List[PIL.Image.Image]] = None,
        input_image_latents: Optional[List[torch.Tensor]] = None, # <- INJECTED LATE FUSION LATENTS
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_pixels: int = 1024 * 1024,
        max_input_image_side_length: int = 1024,
        align_res: bool = True,
        num_inference_steps: int = 28,
        text_guidance_scale: float = 4.0,
        image_guidance_scale: float = 1.0,
        cfg_range: Tuple[float, float] = (0.0, 1.0),
        attention_kwargs: Optional[Dict[str, Any]] = None,
        timesteps: List[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        verbose: bool = False,
        step_func=None,
        **kwargs
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._text_guidance_scale = text_guidance_scale
        self._image_guidance_scale = image_guidance_scale
        self._cfg_range = cfg_range
        self._attention_kwargs = attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt (Early Fusion)
        if prompt_embeds is None:
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self.encode_prompt(
                prompt,
                input_images,
                self.text_guidance_scale > 1.0,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                max_sequence_length=max_sequence_length,
                use_text_encoder_penultimate_layer_feats=use_text_encoder_penultimate_layer_feats
            )

        dtype = self.vae.dtype

        # 4. Prepare control image (Late Fusion Override)
        if input_image_latents is not None:
            ref_latents = input_image_latents # Use precomputed / extracted custom latents
        else:
            ref_latents = self.prepare_image(
                images=input_images,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                max_pixels=max_pixels,
                max_side_length=max_input_image_side_length,
                device=device,
                dtype=dtype,
            )

        if input_images is None and input_image_latents is None:
            input_images_check = []
        elif input_images is not None:
            input_images_check = input_images
        else:
            input_images_check = [None] # Mock length if pure latents passed

        if len(input_images_check) == 1 and align_res and ref_latents is not None:
            try:
                lat_tensor = ref_latents[0][0]
                width, height = lat_tensor.shape[-1] * self.vae_scale_factor, lat_tensor.shape[-2] * self.vae_scale_factor
                ori_width, ori_height = width, height
            except:
                ori_width, ori_height = width, height
        else:
            ori_width, ori_height = width, height
            cur_pixels = height * width
            ratio = (max_pixels / cur_pixels) ** 0.5
            ratio = min(ratio, 1.0)
            height, width = int(height * ratio) // 16 * 16, int(width * ratio) // 16 * 16

        if len(input_images_check) == 0:
            self._image_guidance_scale = 1

        # 5. Prepare generation latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
            self.transformer.config.axes_dim_rope,
            self.transformer.config.axes_lens,
            theta=10000,
        )

        image = self.processing(
            latents=latents,
            ref_latents=ref_latents,
            prompt_embeds=prompt_embeds,
            freqs_cis=freqs_cis,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            device=device,
            dtype=dtype,
            verbose=verbose,
            step_func=step_func,
            **kwargs,
        )

        image = F.interpolate(image, size=(ori_height, ori_width), mode='bilinear')
        image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return image
        else:
            return FMPipelineOutput(images=image)
