FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base
RUN apt-get update && apt-get install -y cuda-nvcc-12-4
RUN nvcc --version
# Install libGL.so.1
# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 
#ENV CUDA_HOME=/usr/local/cuda
#ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget
RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt-get install -y ffmpeg 
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install ComfyUI 
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124 \
    && pip install -r requirements.txt
# 安装 git-lfs
RUN apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install     
WORKDIR /comfyui/custom_nodes/
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
WORKDIR /comfyui/custom_nodes/ComfyUI-VideoHelperSuite
RUN pip install -r requirements.txt
WORKDIR /comfyui/models/
RUN pip install -U modelscope
#Multitalk
WORKDIR /comfyui/custom_nodes/
RUN git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git
RUN git clone https://github.com/chflame163/ComfyUI_LayerStyle.git
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git
RUN git clone https://github.com/christian-byrne/audio-separation-nodes-comfyui.git
RUN git clone -b multitalk https://github.com/kijai/ComfyUI-WanVideoWrapper.git
RUN git clone https://github.com/crystian/ComfyUI-Crystools.git
WORKDIR /comfyui/custom_nodes/ComfyUI_LayerStyle
RUN pip install -r requirements.txt
#WORKDIR /comfyui/custom_nodes/KJNodes
#RUN pip install -r requirements.txt
WORKDIR /comfyui/custom_nodes/audio-separation-nodes-comfyui
RUN pip install -r requirements.txt
WORKDIR /comfyui/custom_nodes/ComfyUI-WanVideoWrapper
RUN pip install -r requirements.txt
WORKDIR /comfyui/custom_nodes/ComfyUI-Crystools
RUN pip install -r requirements.txt
#Multitalk
RUN wget -P /comfyui/models/lora https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors 
RUN wget -P /comfyui/models/diffusion_models https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/WanVideo_2_1_Multitalk_14B_fp8_e4m3fn.safetensors 
RUN wget -P /comfyui/models/vae https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors
RUN wget -P /comfyui/models/diffusion_models https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors 
RUN wget -P /comfyui/models/text_encoders https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors 
RUN wget -P /comfyui/models/controlnet https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_Uni3C_controlnet_fp16.safetensors
# 添加脚本并执行一次
COPY preload_indextts.py /comfyui/
#RUN python3 /comfyui/preload_indextts.py
# 提前 warmup
COPY preload_warmup.py /comfyui/
#RUN python3 /comfyui/preload_warmup.py
# 字体缓存
#RUN python3 -c "from matplotlib import font_manager; font_manager._rebuild()"
WORKDIR /comfyui
# Install runpod
RUN pip install runpod requests
WORKDIR /comfyui/input
RUN wget https://comfyuiyihuan.oss-cn-hangzhou.aliyuncs.com/bd3d3f9b-ce6c-435e-9555-13407f59d7e7.mp3
# python 加速 flag
ENV PYTHONOPTIMIZE=2
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_VERBOSITY=error
# Go back to the root
WORKDIR /
# Add the start and the handler
ADD src/start.sh src/rp_handler.py ./
RUN chmod +x /start.sh

# Start the container
CMD /start.sh
