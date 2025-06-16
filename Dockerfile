# Use Nvidia CUDA base image
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base
RUN apt-get update && apt-get install -y cuda-nvcc-12-4
RUN nvcc --version
#FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 as base
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
# Clean up to reduce image size
# RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt-get install -y ffmpeg 
# Clone ComfyUI repository
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
#RUN git clone https://github.com/WASasquatch/was-node-suite-comfyui.git custom_nodes/was-node-suite-comfyui    
#WORKDIR /comfyui/custom_nodes/was-node-suite-comfyui
#RUN pip install -r requirements.txt
WORKDIR /comfyui/custom_nodes/
RUN git clone https://github.com/unrulpkk/comfyuifunaudiollmv3.git
WORKDIR /comfyui/custom_nodes/comfyuifunaudiollmv3
RUN pip install -r requirements.txt
WORKDIR /comfyui/custom_nodes/
RUN git clone https://github.com/smthemex/ComfyUI_Sonic.git
WORKDIR /comfyui/custom_nodes/ComfyUI_Sonic
RUN pip install -r requirements.txt
WORKDIR /comfyui/models/
RUN mkdir sonic
WORKDIR /comfyui/models/sonic
RUN wget "https://drive.usercontent.google.com/download?id=13HpficBvlmNvTv6W4Oa7agWyzmvmofB4&export=download&authuser=0&confirm=t&uuid=b5068dc9-4c31-454d-a914-149ec47d0fd3&at=ALoNOgndgoQxwIXUZI4peuv08WtG:1749626831748"
RUN wget "https://drive.usercontent.google.com/download?id=1RHWasbgUWZg-mFaQhDJtF1KhpUSecC5d&export=download&authuser=0&confirm=t&uuid=e0f22e45-1598-4ecb-af4a-ee87825fc6e6&at=ALoNOglMjFF4Ta2DVnukpFI5mq_e:1749626929930"
RUN wget "https://drive.usercontent.google.com/download?id=1mjIqU-c5q3qMI74XZd3UrkZek0IDTUUh&export=download&authuser=0&confirm=t&uuid=118a0aec-2ce7-418e-80a9-597c208e4ede&at=ALoNOgn_x2IOwtHF_yo8qtA8JMkm:1749626934470"
RUN wget "https://drive.usercontent.google.com/download?id=1vUYb5NMvDA2XsxRZcB3nF3u1trOtK53h&export=download&authuser=0&confirm=t&uuid=fd882ec7-b586-4a5e-9365-dda43531dc42&at=ALoNOgkJnORsI3QcAhF2L16YzvZ6:1749626932226"
RUN mkdir RIFE
RUN mkdir whisper-tiny
WORKDIR /comfyui/models/sonic/RIFE
RUN wget "https://drive.usercontent.google.com/download?id=1UnSds5DhPRZu4C23I4uOmmahH0J3Dkwl&export=download&authuser=0&confirm=t&uuid=0c7343af-53c9-415f-b2d9-402bc8a483be&at=ALoNOglIV24g0KMT3RptJxbmi3dQ:1749627103585"
WORKDIR /comfyui/models/sonic/whisper-tiny
# 安装 huggingface_hub CLI 工具（如果未安装）
RUN pip install --no-cache-dir huggingface_hub
# 下载指定模型文件到目标目录
RUN huggingface-cli download openai/whisper-tiny \
    --local-dir /comfyui/models/sonic/whisper-tiny \
    --local-dir-use-symlinks False \
    --repo-type model \
    --include "model.safetensors" "config.json" "preprocessor_config.json"
WORKDIR /comfyui/models/
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
WORKDIR /comfyui/models/ComfyUI-VideoHelperSuite
RUN pip install -r requirements.txt
RUN pip install -U modelscope
WORKDIR /
ADD modelscopedown.py ./
RUN python3 modelscopedown.py
#RUN modelscope download cjc1887415157/stable-video-diffusion-img2vid-xt-1-1 --local-dir /comfyui/models/checkpoints
#安装indextts
#WORKDIR /comfyui/custom_nodes/
#RUN git clone https://github.com/chenpipi0807/ComfyUI-Index-TTS.git
#WORKDIR /comfyui/models/
#RUN mkdir IndexTTS-1.5
#RUN huggingface-cli download IndexTeam/IndexTTS-1.5 --local-dir /comfyui/models/IndexTTS-1.5
#WORKDIR /comfyui/custom_nodes/ComfyUI-Index-TTS
#RUN pip install -r requirements.txt
#RUN pip show transformers torch
WORKDIR /comfyui
# Install runpod
RUN pip install runpod requests
# RUN git clone https://github.com/zhilengjun/ComfyUI-FunAudioLLM_V2.git custom_nodes/ComfyUI-FunAudioLLM_V2

# RUN git clone https://github.com/WASasquatch/was-node-suite-comfyui.git custom_nodes/was-node-suite-comfyui


# WORKDIR /comfyui/custom_nodes/ComfyUI-FunAudioLLM_V2
# RUN pip install --no-cache-dir -r requirements.txt
# WORKDIR /comfyui/custom_nodes/was-node-suite-comfyui
# RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /comfyui

RUN git clone https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B.git  models/CosyVoice/CosyVoice2-0.5B

WORKDIR /comfyui/input
RUN wget https://comfyuiyihuan.oss-cn-hangzhou.aliyuncs.com/bd3d3f9b-ce6c-435e-9555-13407f59d7e7.mp3

# Go back to the root
WORKDIR /
# Add the start and the handler
ADD src/start.sh src/rp_handler.py ./
RUN chmod +x /start.sh

# Start the container
CMD /start.sh
