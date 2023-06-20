# syntax=docker/dockerfile:1

# Installing neccessary packages
FROM python:3.9-slim as base

WORKDIR /app
COPY . .

ARG INSTALL_CUDA=false
ARG MODELS="all"
ARG PORT=8080
ARG DATA="examples"

# Setup cuda and base
FROM nvidia/cuda:11.3-base as cuda
FROM base as conditional

RUN if [ "$INSTALL_CUDA" = "true" ]; then \
        echo "Installing CUDA 11.3 altogether before installing PyTorch with CUDA support" \
        apt-get update && apt-get install -y --no-install-recommends \
        cuda-compiler-11-3 \
        cuda-libraries-11-3 \
        cuda-libraries-dev-11-3 \
        && rm -rf /var/lib/apt/lists/* \
        pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113; \
    else \
        nvcc_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1) && \
        if [ -n "$nvcc_version" ]; then \
            echo "CUDA detected (version $nvcc_version), installing PyTorch with CUDA support"; \
            pip install torch==1.12.1+cu$(echo "$nvcc_version" | tr -d .) torchvision==0.13.1+cu$(echo "$nvcc_version" | tr -d .) torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu$(echo "$nvcc_version" | tr -d .); \
        else \
            echo "CUDA not detected, installing PyTorch for CPU use"; \
            pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu; \
    fi

# Stage 3: Final stage
FROM base as final

# Installing requirements
RUN pip install --no-cache-dir -r requirements.txt \
    export interactivenet_results="/app/apps/interactivenet/model"

RUN if [ $MODELS != "all" ]; then \
    download_and_install_model -t $MODELS \
    monailabel start_server --app apps/interactivenet --studies $DATA --conf models $MODELS