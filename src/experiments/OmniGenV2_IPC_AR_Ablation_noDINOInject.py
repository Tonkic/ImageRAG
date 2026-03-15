from _ipc_ar_ablation_common import exec_ablation


if __name__ == "__main__":
    exec_ablation(__file__, [
        (
            'out_root = f"results/{args.retrieval_method}/{dt.strftime(\'%Y.%-m.%-d\')}/OmniGenV2_IPC_AR_{dt.strftime(\'%H-%M-%S\')}"',
            'out_root = f"results/{args.retrieval_method}/{dt.strftime(\'%Y.%-m.%-d\')}/OmniGenV2_IPC_AR_noDINOInject_{dt.strftime(\'%H-%M-%S\')}"',
        ),
        (
            'ref_image_path=best_ref, dino_extractor=dino_extractor_global,',
            'ref_image_path=None, dino_extractor=None,',
        ),
        (
            'ref_image_path=cur_ref, dino_extractor=dino_extractor_global,',
            'ref_image_path=None, dino_extractor=None,',
        ),
        (
            'cur_lambda = min(cur_lambda + args.dino_lambda_step, args.dino_lambda_max)',
            'cur_lambda = args.dino_lambda_init  # ABLATION: disable adaptive lambda',
        ),
    ])
