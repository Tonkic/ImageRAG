from _ipc_ar_ablation_common import exec_ablation


if __name__ == "__main__":
    exec_ablation(__file__, [
        (
            'out_root = f"results/{args.retrieval_method}/{dt.strftime(\'%Y.%-m.%-d\')}/OmniGenV2_IPC_AR_{dt.strftime(\'%H-%M-%S\')}"',
            'out_root = f"results/{args.retrieval_method}/{dt.strftime(\'%Y.%-m.%-d\')}/OmniGenV2_IPC_AR_noIPCCues_{dt.strftime(\'%H-%M-%S\')}"',
        ),
        (
            'if not mismatch_cues:',
            'if False:  # ABLATION: ignore IPC cues as reinforcement input',
        ),
        (
            'if diag.get("mismatch_cues"):',
            'if True:  # ABLATION: keep reinforce but remove IPC cues',
        ),
        (
            'f_log.write(f"  [Step5-A] LLM reinforce; IPC_cues={diag.get(\'mismatch_cues\',[])[:3]}\\n")',
            'f_log.write("  [Step5-A] LLM reinforce; IPC_cues=DISABLED\\n")',
        ),
        (
            'Identity mismatch diagnostics from IPC: {cues_str or \'(none)\'}.',
            'Target entity to emphasize: {entity_name}.',
        ),
        (
            'Rewrite the prompt to STRONGLY EMPHASIZE identity-corrective cues:',
            'Rewrite the prompt to STRONGLY EMPHASIZE the target entity identity:',
        ),
        (
            'mismatch_cues=diag.get("mismatch_cues", []),',
            'mismatch_cues=[],',
        ),
    ])
