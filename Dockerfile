# Use Nvidia CUDA base image
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base
RUN apt-get update && apt-get install -y cuda-nvcc-12-4
RUN apt-get install -y cuda-nvcc-12-4 cuda-toolkit-12-4 && \
    ln -s /usr/local/cuda/extras/CUPTI/include/cuda_profiler_api.h /usr/local/cuda/include/cuda_profiler_api.h
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
WORKDIR /comfyui/custom_nodes    
RUN git clone https://github.com/WASasquatch/was-node-suite-comfyui.git
WORKDIR /comfyui/custom_nodes/was-node-suite-comfyui
RUN pip install -r requirements.txt
WORKDIR /comfyui/custom_nodes/
RUN git clone https://github.com/unrulpkk/comfyuifunaudiollmv3.git
WORKDIR /comfyui/custom_nodes/comfyuifunaudiollmv3
RUN pip install -r requirements.txt
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
WORKDIR /comfyui/custom_nodes/ComfyUI-Custom-Scripts
RUN pip install -r requirements.txt
WORKDIR /comfyui/custom_nodes/ComfyUI_LayerStyle
RUN pip install -r requirements.txt
WORKDIR /comfyui/custom_nodes/KJNodes
RUN pip install -r requirements.txt
WORKDIR /comfyui/custom_nodes/audio-separation-nodes-comfyui
RUN pip install -r requirements.txt
WORKDIR /comfyui/custom_nodes/ComfyUI-WanVideoWrapper
RUN pip install -r requirements.txt
#Multitalk
RUN wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors -O /comfyui/models/lora/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors
RUN wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/WanVideo_2_1_Multitalk_14B_fp8_e4m3fn.safetensors -O /comfyui/models/checkpoints/WanVideo_2_1_Multitalk_14B_fp8_e4m3fn.safetensors 
RUN wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors -O /comfyui/models/vae/Wan2_1_VAE_bf16.safetensors
RUN wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors -O /comfyui/models/checkpoints/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors
RUN wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors -O /comfyui/models/checkpoints/umt5-xxl-enc-fp8_e4m3fn.safetensors 
RUN wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_Uni3C_controlnet_fp16.safetensors -O /comfyui/models/controlnet/Wan21_Uni3C_controlnet_fp16.safetensors
#Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors
#WanVideo_2_1_Multitalk_14B_fp8_e4m3fn.safetensors
#Wan2_1_VAE_bf16.safetensors
#Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors
#umt5-xxl-enc-fp8_e4m3fn.safetensors
#Wan21_Uni3C_controlnet_fp16.safetensors 
#WORKDIR /comfyui/models/
#WORKDIR /
#ADD modelscopedown.py ./
#RUN python3 modelscopedown.py
#RUN modelscope download cjc1887415157/stable-video-diffusion-img2vid-xt-1-1 --local-dir /comfyui/models/checkpoints
#安装indextts
WORKDIR /comfyui/custom_nodes/
RUN git clone https://github.com/chenpipi0807/ComfyUI-Index-TTS.git
WORKDIR /comfyui/models/
RUN mkdir IndexTTS-1.5
RUN huggingface-cli download IndexTeam/IndexTTS-1.5 --local-dir /comfyui/models/IndexTTS-1.5
WORKDIR /comfyui/custom_nodes/ComfyUI-Index-TTS
RUN pip install -r requirements.txt
WORKDIR /comfyui/custom_nodes/ComfyUI-Index-TTS
RUN TORCH_CUDA_ARCH_LIST="7.5;8.0;8.9" \
    python3 setup.py build_ext --inplace || true
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
