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
RUN apt-get install -y git-lfs && \
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
WORKDIR /comfyui/custom_nodes    
RUN git clone https://github.com/WASasquatch/was-node-suite-comfyui.git
WORKDIR /comfyui/custom_nodes/was-node-suite-comfyui
RUN pip install -r requirements.txt
#安装indextts
WORKDIR /comfyui/custom_nodes/
WORKDIR /comfyui
# Install runpod
RUN pip install runpod requests
RUN pip install oss2
RUN pip install sageattention
WORKDIR /comfyui/input
RUN wget https://comfyuiyihuan.oss-cn-hangzhou.aliyuncs.com/bd3d3f9b-ce6c-435e-9555-13407f59d7e7.mp3
# Go back to the root
WORKDIR /
# Add the start and the handler
ADD src/start.sh src/rp_handler.py ./
RUN chmod +x /start.sh

# Start the container
CMD /start.sh
