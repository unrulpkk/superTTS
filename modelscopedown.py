from modelscope.hub.snapshot_download import snapshot_download

snapshot_download(
    "cjc1887415157/stable-video-diffusion-img2vid-xt-1-1",
    cache_dir="/comfyui/models/checkpoints"
)
