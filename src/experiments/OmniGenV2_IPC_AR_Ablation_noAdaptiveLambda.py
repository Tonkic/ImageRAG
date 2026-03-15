from _ipc_ar_ablation_common import exec_ablation


if __name__ == "__main__":
    exec_ablation(__file__, [
        (
            'out_root = f"results/{args.retrieval_method}/{dt.strftime(\'%Y.%-m.%-d\')}/OmniGenV2_IPC_AR_{dt.strftime(\'%H-%M-%S\')}"',
            'out_root = f"results/{args.retrieval_method}/{dt.strftime(\'%Y.%-m.%-d\')}/OmniGenV2_IPC_AR_noAdaptiveLambda_{dt.strftime(\'%H-%M-%S\')}"',
        ),
        (
            'cur_lambda = min(cur_lambda + args.dino_lambda_step, args.dino_lambda_max)',
            'cur_lambda = args.dino_lambda_init  # ABLATION: keep lambda fixed',
        ),
    ])