from _ipc_ar_ablation_common import exec_ablation


if __name__ == "__main__":
    exec_ablation(__file__, [
        (
            'out_root = f"results/{args.retrieval_method}/{dt.strftime(\'%Y.%-m.%-d\')}/OmniGenV2_IPC_AR_{dt.strftime(\'%H-%M-%S\')}"',
            'out_root = f"results/{args.retrieval_method}/{dt.strftime(\'%Y.%-m.%-d\')}/OmniGenV2_IPC_AR_noStage2Rerank_{dt.strftime(\'%H-%M-%S\')}"',
        ),
        (
            'for idx,(cp,cs) in enumerate(zip(candidates,cscores)):\n\t\ttry:\n\t\t\timg64 = encode_image(cp)\n\t\t\tif not img64: continue\n\t\t\tresp = vl_client.chat.completions.create(\n\t\t\t\tmodel=vl_model,\n\t\t\t\tmessages=[{"role":"user","content":[\n\t\t\t\t\t{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img64}"}},\n\t\t\t\t\t{"type":"text","text":f"Does this show \'{entity}\'? Answer YES or NO only."}]}],\n\t\t\t\ttemperature=0.01, max_tokens=8)\n\t\t\tans = (resp.choices[0].message.content or "").strip().upper()\n\t\t\tif "YES" in ans:\n\t\t\t\tdetails["stage_b_passed"] += 1\n\t\t\t\tvalid_refs.append(cp)\n\t\t\t\tif len(valid_refs)==1: best_score = cs\n\t\t\t\tlog(f"    [#{idx}] PASS ✓")\n\t\t\telse:\n\t\t\t\tlog(f"    [#{idx}] FAIL ✗")\n\t\texcept Exception as e:\n\t\t\tlog(f"    [#{idx}] ERR: {e}")\n\n\treturn valid_refs, best_score, details\n',
            'for idx,(cp,cs) in enumerate(zip(candidates,cscores)):\n\t\tif idx == 0:\n\t\t\tvalid_refs.append(cp)\n\t\t\tbest_score = cs\n\t\t\tdetails["stage_b_passed"] = 1\n\t\t\tlog(f"    [#{idx}] PASS ✓ (Stage-1 top-1, no rerank)")\n\t\telse:\n\t\t\tlog(f"    [#{idx}] SKIP (no stage-2 rerank)")\n\n\treturn valid_refs, best_score, details\n',
        ),
    ])