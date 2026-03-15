from _tac_slim_ablation_common import exec_ablation


if __name__ == "__main__":
    exec_ablation(
        __file__,
        [
            (
                'out_root  = f"results/{args.retrieval_method}/{timestamp}/OmniGenV2_TACAttr_AR_{run_time}"',
                'out_root  = f"results/{args.retrieval_method}/{timestamp}/OmniGenV2_TACAttr_noDINO_AR_{run_time}"',
            ),
            (
                '                ref_image_path=best_ref, dino_extractor=dino_extractor_global,',
                '                ref_image_path=best_ref, dino_extractor=None,',
            ),
            (
                '                        ref_image_path=cur_ref, dino_extractor=dino_extractor_global,',
                '                        ref_image_path=cur_ref, dino_extractor=None,',
            ),
        ],
    )
