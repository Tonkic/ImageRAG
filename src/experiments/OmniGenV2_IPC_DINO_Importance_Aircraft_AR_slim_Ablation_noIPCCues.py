from _ipc_slim_ablation_common import exec_ablation


if __name__ == "__main__":
    exec_ablation(__file__, [
        (
            'out_root = f"results/{args.retrieval_method}/{dt.strftime(\'%Y.%-m.%-d\')}/OmniGenV2_IPC_AR_{dt.strftime(\'%H-%M-%S\')}"',
            'out_root = f"results/{args.retrieval_method}/{dt.strftime(\'%Y.%-m.%-d\')}/OmniGenV2_IPC_AR_noIPCCues_{dt.strftime(\'%H-%M-%S\')}"',
        ),
        (
            'if tifa_use and (failed_features or diag.get("mismatch_cues")):',
            'if tifa_use and failed_features:',
        ),
        (
            '                            mismatch_cues=diag.get("mismatch_cues", []),',
            '                            mismatch_cues=[],',
        ),
    ])
