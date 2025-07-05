# preload_indextts.py
from indextts.models import IndexTTSModel

model_path = "/comfyui/models/IndexTTS-1.5"
model = IndexTTSModel.load_from_folder(model_path, device="cuda", fp16=True, use_cuda_kernel=True)
print("IndexTTS 模型 preload 完成")
