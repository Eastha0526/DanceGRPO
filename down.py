from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="Qwen/Qwen-Image",
    local_dir="./data/qwenimage",
    local_dir_use_symlinks=False
)

snapshot_download(
    repo_id="xswu/HPSv2",
    local_dir="./hps_ckpt",
    local_dir_use_symlinks=False
)

snapshot_download(
    repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    local_dir="./hps_ckpt",
    local_dir_use_symlinks=False
)
