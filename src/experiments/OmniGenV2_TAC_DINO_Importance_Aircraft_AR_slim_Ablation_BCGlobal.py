from _tac_slim_ablation_common import exec_ablation


if __name__ == "__main__":
    exec_ablation(
        __file__,
        [
            (
                'out_root  = f"results/{args.retrieval_method}/{timestamp}/OmniGenV2_TACAttr_AR_{run_time}"',
                'out_root  = f"results/{args.retrieval_method}/{timestamp}/OmniGenV2_BCGlobal_AR_{run_time}"',
            ),
            (
                '        f.write("=== Dual-Client Attribute-TAC Pipeline ===\\n")',
                '        f.write("=== Dual-Client BC-Global Pipeline ===\\n")',
            ),
            (
                '        f.write("=== Dual-Client Attribute-TAC Pipeline Summary ===\\n")',
                '        f.write("=== Dual-Client BC-Global Pipeline Summary ===\\n")',
            ),
            (
                '            f_log.write(">>> STEP 4: Attribute-based TAC\\n")',
                '            f_log.write(">>> STEP 4: Binary Critic Global YES/NO\\n")',
            ),
            (
                '            diagnosis = tac_attribute_diagnosis(',
                '            diagnosis = binary_critic_diagnosis(',
            ),
            (
                '                    diagnosis = tac_attribute_diagnosis(',
                '                    diagnosis = binary_critic_diagnosis(',
            ),
            (
                '            f.write(f"  Attribute-TAC accept rate: {warmup[\'accepts\']}/{bc} ({warmup[\'accepts\']/bc:.2%})\\n")',
                '            f.write(f"  BC-Global accept rate: {warmup[\'accepts\']}/{bc} ({warmup[\'accepts\']/bc:.2%})\\n")',
            ),
        ],
    )
