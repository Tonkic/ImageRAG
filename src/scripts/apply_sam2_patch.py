import os

file_path = r"e:\ImageRAG-main\src\experiments\OmniGenV2_TAC_DINO_Importance_Aircraft.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Target 1: apply_sam2_matting
t1_old = '''    def to(self, device):
        """Move model to specified device."""
        if self.model is not None:
            self.model.to(device)
            self.device = device if isinstance(device, str) else str(device)
        return self


# ==============================================================================
# Step 1: Importance-Aware Input Interpreter (非对称解析)
# =============================================================================='''
t1_new = '''    def to(self, device):
        """Move model to specified device."""
        if self.model is not None:
            self.model.to(device)
            self.device = device if isinstance(device, str) else str(device)
        return self


# ==============================================================================
# SAM2 Physical Matting (For DINO pureness only)
# ==============================================================================
def apply_sam2_matting(image_path, output_path, device="cuda"):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import numpy as np

        sam2_checkpoint = "sam2/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        predictor.set_image(image_np)

        H, W, _ = image_np.shape
        input_point = np.array([[W//2, H//2]])
        input_label = np.array([1])

        masks, _, _ = predictor.predict(
            point_coords=input_point, point_labels=input_label, multimask_output=False,
        )

        mask = masks[0]
        image_np[~mask] = [0, 0, 0] # 背景涂黑

        Image.fromarray(image_np).save(output_path)
        del sam2_model, predictor
        import torch
        torch.cuda.empty_cache()
        return output_path
    except Exception as e:
        print(f"[SAM2] Matting failed: {e}")
        return image_path


# ==============================================================================
# Step 1: Importance-Aware Input Interpreter (非对称解析)
# =============================================================================='''
content = content.replace(t1_old, t1_new)

# Target 2: dino_injected_generation signature
t2_old = '''def dino_injected_generation(pipe, generation_prompt, ref_image_path,
                              dino_extractor, dino_lambda,
                              seed, omnigen_device,
                              output_path, f_log=None,
                              height=512, width=512,
                              img_guidance_scale=1.6,
                              text_guidance_scale=2.5,
                              negative_prompt=""):'''
t2_new = '''def dino_injected_generation(pipe, generation_prompt, ref_image_path,
                              dino_extractor, dino_lambda,
                              seed, omnigen_device,
                              output_path, f_log=None,
                              height=512, width=512,
                              img_guidance_scale=1.6,
                              text_guidance_scale=2.5,
                              negative_prompt="",
                              decouple_threshold=0.25,
                              dino_ref_path=None):'''
content = content.replace(t2_old, t2_new)

# Target 3: target_dino_path logic
t3_old = '''    if dino_extractor is not None and ref_image_path is not None:
        try:
            patch_features, cls_token = dino_extractor.extract_patch_features(ref_image_path)
            if cls_token is not None:'''
t3_new = '''    target_dino_path = dino_ref_path if dino_ref_path else ref_image_path
    if dino_extractor is not None and target_dino_path is not None:
        try:
            patch_features, cls_token = dino_extractor.extract_patch_features(target_dino_path)
            if cls_token is not None:'''
content = content.replace(t3_old, t3_new)

# Target 4: pipe() calls
t4_old = '''    try:
        result = pipe(
            prompt=gen_prompt,
            input_images=input_images if input_images else None,
            height=height,
            width=width,
            text_guidance_scale=text_guidance_scale,
            image_guidance_scale=effective_img_guidance,
            num_inference_steps=50,
            generator=gen,
            negative_prompt=negative_prompt,
        )
        result.images[0].save(output_path)
        log(f"  [Step3] Image saved to {output_path}")

    except Exception as e:
        log(f"  [Step3] Generation error: {e}")
        # Fallback: generate without reference
        try:
            gen = torch.Generator("cuda").manual_seed(seed)
            result = pipe(
                prompt=generation_prompt,
                input_images=None,
                height=height, width=width,
                text_guidance_scale=text_guidance_scale,
                image_guidance_scale=1.0,
                num_inference_steps=50,
                generator=gen,
                negative_prompt=negative_prompt,
            )
            result.images[0].save(output_path)'''
t4_new = '''    try:
        result = pipe(
            prompt=gen_prompt,
            input_images=input_images if input_images else None,
            height=height,
            width=width,
            text_guidance_scale=text_guidance_scale,
            image_guidance_scale=effective_img_guidance,
            num_inference_steps=50,
            generator=gen,
            negative_prompt=negative_prompt,
            decouple_threshold=decouple_threshold,
        )
        result.images[0].save(output_path)
        log(f"  [Step3] Image saved to {output_path}")

    except Exception as e:
        log(f"  [Step3] Generation error: {e}")
        # Fallback: generate without reference
        try:
            gen = torch.Generator("cuda").manual_seed(seed)
            result = pipe(
                prompt=generation_prompt,
                input_images=None,
                height=height, width=width,
                text_guidance_scale=text_guidance_scale,
                image_guidance_scale=1.0,
                num_inference_steps=50,
                generator=gen,
                negative_prompt=negative_prompt,
                decouple_threshold=decouple_threshold,
            )
            result.images[0].save(output_path)'''
content = content.replace(t4_old, t4_new)

# Target 5: reflexive_regeneration signature
t5_old = '''def reflexive_regeneration(pipe, diagnosis, interpretation,
                           ref_image_path, dino_extractor,
                           current_dino_lambda, retry_idx,
                           seed, omnigen_device, output_path,
                           base_negative_prompt="",
                           height=512, width=512,
                           img_guidance_scale=1.6,
                           text_guidance_scale=2.5,
                           dino_lambda_step=0.15,
                           dino_lambda_max=0.8,
                           f_log=None):'''
t5_new = '''def reflexive_regeneration(pipe, diagnosis, interpretation,
                           ref_image_path, dino_extractor,
                           current_dino_lambda, retry_idx,
                           seed, omnigen_device, output_path,
                           base_negative_prompt="",
                           height=512, width=512,
                           img_guidance_scale=1.6,
                           text_guidance_scale=2.5,
                           dino_lambda_step=0.15,
                           dino_lambda_max=0.8,
                           decouple_threshold=0.25,
                           dino_ref_path=None,
                           f_log=None):'''
content = content.replace(t5_old, t5_new)

# Target 6: reflexive_regeneration downcall
t6_old = '''    effective_guidance = dino_injected_generation(
        pipe=pipe,
        generation_prompt=refined_prompt,
        ref_image_path=ref_image_path,
        dino_extractor=dino_extractor,
        dino_lambda=next_dino_lambda,
        seed=seed + retry_idx + 1,
        omnigen_device=omnigen_device,
        output_path=output_path,
        f_log=f_log,
        height=height,
        width=width,
        img_guidance_scale=img_guidance_scale,
        text_guidance_scale=text_guidance_scale,
        negative_prompt=enhanced_negative,
    )'''
t6_new = '''    effective_guidance = dino_injected_generation(
        pipe=pipe,
        generation_prompt=refined_prompt,
        ref_image_path=ref_image_path,
        dino_extractor=dino_extractor,
        dino_lambda=next_dino_lambda,
        seed=seed + retry_idx + 1,
        omnigen_device=omnigen_device,
        output_path=output_path,
        f_log=f_log,
        height=height,
        width=width,
        img_guidance_scale=img_guidance_scale,
        text_guidance_scale=text_guidance_scale,
        negative_prompt=enhanced_negative,
        decouple_threshold=decouple_threshold,
        dino_ref_path=dino_ref_path,
    )'''
content = content.replace(t6_old, t6_new)

# Target 7: Step 2.5 Hook
t7_old = '''            # ==================================================================
            # STEP 3: Initial Generation (DINO-Injected / 结构先验注入)
            # ==================================================================
            f_log.write(">>> STEP 3: Initial Generation (DINO Structure Prior Injection)\\n")

            v1_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V1.png")
            current_dino_lambda = args.dino_lambda_init

            effective_guidance = dino_injected_generation(
                pipe=pipe,
                generation_prompt=generation_prompt,
                ref_image_path=best_ref,
                dino_extractor=dino_extractor_global,
                dino_lambda=current_dino_lambda,
                seed=args.seed,
                omnigen_device=omnigen_device,
                output_path=v1_path,
                f_log=f_log,
                height=args.height,
                width=args.width,
                img_guidance_scale=args.image_guidance_scale,
                text_guidance_scale=args.text_guidance_scale,
                negative_prompt=args.negative_prompt,
            )'''
t7_new = '''            # ------------------------------------------------------------------
            # [新增] STEP 2.5: SAM2 Physical Matting (物理隔离)
            # ------------------------------------------------------------------
            dino_ref_image = best_ref  # 默认 DINO 也用原图

            if args.use_sam2_matting and best_ref is not None:
                f_log.write(">>> STEP 2.5: SAM2 Physical Matting for DINO\\n")
                clean_ref_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_clean_ref.png")
                dino_ref_image = apply_sam2_matting(best_ref, clean_ref_path, device=retrieval_device)
                f_log.write(f"  Clean ref for DINO saved to: {dino_ref_image}\\n\\n")

            # ==================================================================
            # STEP 3: Initial Generation (DINO-Injected / 结构先验注入)
            # ==================================================================
            f_log.write(">>> STEP 3: Initial Generation (DINO Structure Prior Injection)\\n")

            v1_path = os.path.join(DATASET_CONFIG['output_path'], f"{safe_name}_V1.png")
            current_dino_lambda = args.dino_lambda_init

            effective_guidance = dino_injected_generation(
                pipe=pipe,
                generation_prompt=generation_prompt,
                ref_image_path=best_ref,         # 👈 给 OmniGen2 依然传入带背景的原图！
                dino_ref_path=dino_ref_image,    # 👈 [新增传参] 给 DINO 传入抠黑背景的图！
                dino_extractor=dino_extractor_global,
                dino_lambda=current_dino_lambda,
                seed=args.seed,
                omnigen_device=omnigen_device,
                output_path=v1_path,
                f_log=f_log,
                height=args.height,
                width=args.width,
                img_guidance_scale=args.image_guidance_scale,
                text_guidance_scale=args.text_guidance_scale,
                negative_prompt=args.negative_prompt,
                decouple_threshold=args.decouple_threshold,
            )'''
content = content.replace(t7_old, t7_new)

# Target 8: Step 5 Hook
t8_old = '''                    # Step 5: Reflexive Re-generation
                    current_dino_lambda = reflexive_regeneration(
                        pipe=pipe,
                        diagnosis=diagnosis,
                        interpretation=interpretation,
                        ref_image_path=best_ref,
                        dino_extractor=dino_extractor_global,
                        current_dino_lambda=current_dino_lambda,
                        retry_idx=retry_cnt,
                        seed=args.seed,
                        omnigen_device=omnigen_device,
                        output_path=next_path,
                        base_negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        img_guidance_scale=args.image_guidance_scale,
                        text_guidance_scale=args.text_guidance_scale,
                        dino_lambda_step=args.dino_lambda_step,
                        dino_lambda_max=args.dino_lambda_max,
                        f_log=f_log,
                    )'''
t8_new = '''                    # Step 5: Reflexive Re-generation
                    current_dino_lambda = reflexive_regeneration(
                        pipe=pipe,
                        diagnosis=diagnosis,
                        interpretation=interpretation,
                        ref_image_path=best_ref,
                        dino_extractor=dino_extractor_global,
                        current_dino_lambda=current_dino_lambda,
                        retry_idx=retry_cnt,
                        seed=args.seed,
                        omnigen_device=omnigen_device,
                        output_path=next_path,
                        base_negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        img_guidance_scale=args.image_guidance_scale,
                        text_guidance_scale=args.text_guidance_scale,
                        dino_lambda_step=args.dino_lambda_step,
                        dino_lambda_max=args.dino_lambda_max,
                        decouple_threshold=args.decouple_threshold,
                        dino_ref_path=dino_ref_image,
                        f_log=f_log,
                    )'''
content = content.replace(t8_old, t8_new)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

# verify all replaced
assert t1_old not in content
assert t2_old not in content
assert t3_old not in content
assert t4_old not in content
assert t5_old not in content
assert t6_old not in content
assert t7_old not in content
assert t8_old not in content
print("All 8 sections successfully replaced.")
