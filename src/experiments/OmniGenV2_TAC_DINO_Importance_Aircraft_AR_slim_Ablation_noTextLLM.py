from _tac_slim_ablation_common import exec_ablation


if __name__ == "__main__":
    exec_ablation(
        __file__,
        [
            (
                'out_root  = f"results/{args.retrieval_method}/{timestamp}/OmniGenV2_TACAttr_AR_{run_time}"',
                'out_root  = f"results/{args.retrieval_method}/{timestamp}/OmniGenV2_TACAttr_noTextLLM_AR_{run_time}"',
            ),
            (
                '        f.write("=== Dual-Client Attribute-TAC Pipeline ===\\n")',
                '        f.write("=== Dual-Client noTextLLM Attribute-TAC Pipeline ===\\n")',
            ),
            (
                '        f.write("=== Dual-Client Attribute-TAC Pipeline Summary ===\\n")',
                '        f.write("=== Dual-Client noTextLLM Attribute-TAC Pipeline Summary ===\\n")',
            ),
            (
                '    # -- Text Client (Qwen3-Omni-30B) --\n    if args.text_api_key:\n        _tc = openai.OpenAI(api_key=args.text_api_key, base_url=args.text_api_base)\n        text_client = UsageTrackingClient(_tc)\n        print(f"[Setup] text_client → {args.text_model} (SiliconFlow API, JSON mode for Step1)")\n    else:\n        print("[Setup] WARNING: --text_api_key not set. Using fallback local VL for text tasks.")\n        text_client = None  # Will be set to vl_client after\n',
                '    # -- Text Client disabled in noTextLLM ablation --\n    text_client = None\n    print("[Setup] text_client disabled (noTextLLM ablation)")\n',
            ),
            (
                '    if text_client is None:\n        text_client = vl_client   # fallback\n    return pipe, text_client, vl_client, vl_model\n',
                '    return pipe, None, vl_client, vl_model\n',
            ),
            (
                """            # ---- STEP 1: Input Interpreter (text_client) ----
            f_log.write(">>> STEP 1: Input Interpreter (LLM API)\\n")
            move_helpers_to_gpu(retrieval_device)
            interp = importance_aware_input_interpreter(prompt, text_client, domain="aircraft")

            if args.warmup_n_classes > 0:
                warmup["classes"] += 1
                if interp.get("_parser_recovery_used"): warmup["input_recovery"] += 1

            entity    = interp["high_importance"]["entity"]
            ret_query = interp["retrieval_query"]
            gen_prompt= interp["generation_prompt"]   # 大模型 thinking 后的 ground truth prompt
            f_log.write(f"  Entity: {entity}\\n  RetQuery: {ret_query}\\n")
            f_log.write(f"  GenPrompt[:200]: {gen_prompt[:200]}\\n\\n")

            # Knowledge specs via text_client
            ref_specs = None
            try:
                # generate_knowledge_specs uses message_gpt internally; we replicate with text_client
                specs_msg = [{"role":"user","content":
                    f"You are an aviation expert. List the top 4 visual identification features of '{class_name}' (engine count/placement, wing config, tail, distinctive features). Plain bullet list only."}]
                ref_specs = call_text_api(text_client, specs_msg, max_tokens=512)
                f_log.write(f"  Specs[:150]: {ref_specs[:150]}\\n\\n")
            except Exception as e:
                f_log.write(f"  Specs error: {e}\\n\\n")
""",
                """            # ---- STEP 1: Rule-based Init (no text LLM) ----
            f_log.write(">>> STEP 1: Rule-based Init (no text LLM)\\n")
            move_helpers_to_gpu(retrieval_device)
            interp = build_rule_based_interpretation(prompt)

            if args.warmup_n_classes > 0:
                warmup["classes"] += 1

            entity    = interp["high_importance"]["entity"]
            ret_query = interp["retrieval_query"]
            gen_prompt= interp["generation_prompt"]
            ref_specs = None
            f_log.write(f"  Entity: {entity}\\n  RetQuery: {ret_query}\\n")
            f_log.write(f"  GenPrompt[:200]: {gen_prompt[:200]}\\n\\n")
""",
            ),
            (
                '            best_ref = valid_refs[0] if valid_refs else None\n            f_log.write(f"  best_ref: {os.path.basename(best_ref) if best_ref else \'None\'}\\n\\n")\n',
                '            best_ref = valid_refs[0] if valid_refs else None\n            ref_specs = generate_vlm_reference_specs(class_name, best_ref, vl_client, vl_model, f_log=f_log, max_items=4)\n            seed_features = collect_tac_features(key_features=[], reference_specs=ref_specs, max_items=4)\n            if seed_features:\n                gen_prompt = rule_reinforce_prompt(gen_prompt, seed_features, entity_name=entity, f_log=f_log)\n            f_log.write(f"  best_ref: {os.path.basename(best_ref) if best_ref else \'None\'}\\n")\n            if ref_specs:\n                f_log.write(f"  VLM Specs[:150]: {ref_specs[:150]}\\n\\n")\n            else:\n                f_log.write("  VLM Specs: None\\n\\n")\n',
            ),
            (
                '            tifa_use = args.max_retries > 2 and bool(key_features)',
                '            tifa_use = False',
            ),
            (
                '            if tifa_use:\n                f_log.write("  [TIFA] Feature-level evaluation:\\n")\n                feature_eval = tifa_feature_eval(\n                    key_features, v1_path, vl_client, vl_model, f_log=f_log)\n                failed_features = [f for f, ok in feature_eval.items() if not ok]\n                f_log.write(f"  [TIFA] failed={failed_features}\\n")\n            else:\n                failed_features = []\n',
                '            failed_features = [f for f in diagnosis.get("missing_attributes", []) if f != "entity_identity"]\n            f_log.write(f"  [TAC] failed={failed_features}\\n")\n',
            ),
            (
                '                    # ---- Step 5-A: LLM Prompt Reinforcement (仅 max_retries > 2) ----\n                    if tifa_use and failed_features:\n                        f_log.write(f"  [Step5-A] LLM prompt reinforcement, failed={failed_features}\\n")\n                        current_gen_prompt = llm_reinforce_prompt(\n                            current_gen_prompt, failed_features, text_client,\n                            entity_name=entity, f_log=f_log)\n',
                '                    # ---- Step 5-A: Rule Prompt Reinforcement (no text LLM) ----\n                    if failed_features:\n                        f_log.write(f"  [Step5-A] Rule prompt reinforcement, failed={failed_features}\\n")\n                        current_gen_prompt = rule_reinforce_prompt(\n                            current_gen_prompt, failed_features, entity_name=entity, f_log=f_log)\n',
            ),
            (
                '                    if tifa_use:\n                        feature_eval = tifa_feature_eval(\n                            key_features, next_path, vl_client, vl_model, f_log=f_log)\n                        failed_features = [f for f, ok in feature_eval.items() if not ok]\n                        f_log.write(f"  [TIFA] re-eval failed={failed_features}\\n")\n',
                '                    failed_features = [f for f in diagnosis.get("missing_attributes", []) if f != "entity_identity"]\n                    f_log.write(f"  [TAC] re-eval failed={failed_features}\\n")\n',
            ),
        ],
    )
