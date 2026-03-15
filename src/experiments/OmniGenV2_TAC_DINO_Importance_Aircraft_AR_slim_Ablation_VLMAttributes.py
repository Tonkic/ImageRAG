from _tac_slim_ablation_common import exec_ablation


if __name__ == "__main__":
    exec_ablation(
        __file__,
        [
            (
                'out_root  = f"results/{args.retrieval_method}/{timestamp}/OmniGenV2_TACAttr_AR_{run_time}"',
                'out_root  = f"results/{args.retrieval_method}/{timestamp}/OmniGenV2_TACAttr_VLMAttributes_AR_{run_time}"',
            ),
            (
                '            # Knowledge specs via text_client\n            ref_specs = None\n            try:\n                # generate_knowledge_specs uses message_gpt internally; we replicate with text_client\n                specs_msg = [{"role":"user","content":\n                    f"You are an aviation expert. List the top 4 visual identification features of \'{class_name}\' (engine count/placement, wing config, tail, distinctive features). Plain bullet list only."}]\n                ref_specs = call_text_api(text_client, specs_msg, max_tokens=512)\n                f_log.write(f"  Specs[:150]: {ref_specs[:150]}\\n\\n")\n            except Exception as e:\n                f_log.write(f"  Specs error: {e}\\n\\n")',
                '            # Attribute source ablation: remove LLM-generated attributes\n            ref_specs = None\n            f_log.write("  Specs: skipped LLM attribute generation (VLM-only ablation)\\n\\n")',
            ),
            (
                '            best_ref = valid_refs[0] if valid_refs else None\n            f_log.write(f"  best_ref: {os.path.basename(best_ref) if best_ref else \'None\'}\\n\\n")',
                '            best_ref = valid_refs[0] if valid_refs else None\n            ref_specs = generate_vlm_reference_specs(class_name, best_ref, vl_client, vl_model, f_log=f_log, max_items=4)\n            f_log.write(f"  best_ref: {os.path.basename(best_ref) if best_ref else \'None\'}\\n")\n            if ref_specs:\n                f_log.write(f"  VLM Specs[:150]: {ref_specs[:150]}\\n\\n")\n            else:\n                f_log.write("  VLM Specs: None\\n\\n")',
            ),
            (
                '            critic_features = collect_tac_features(key_features=key_features, reference_specs=ref_specs, max_items=5)',
                '            critic_features = collect_tac_features(key_features=[], reference_specs=ref_specs, max_items=5)',
            ),
        ],
    )
