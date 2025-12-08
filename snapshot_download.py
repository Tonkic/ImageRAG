from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="FoundationVision/Infinity",
    allow_patterns="infinity_8b_weights/*",  # 只下载权重文件夹
    local_dir="./Infinity"  # 保存路径
)
print(f"模型已下载至: {model_path}")