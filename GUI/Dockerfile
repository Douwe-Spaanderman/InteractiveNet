# syntax=docker/dockerfile:1

# Installing neccessary packages
FROM python:3.9 as base

ARG MODELS=all

WORKDIR /app
COPY . .

# Check if CUDA is available
RUN nvcc_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1) && \
    if [ -n "$nvcc_version" ]; then \
    echo "CUDA detected (version $nvcc_version), installing PyTorch with CUDA support"; \
    pip install torch==1.12.1+cu$(echo "$nvcc_version" | tr -d .) torchvision==0.13.1+cu$(echo "$nvcc_version" | tr -d .) torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu$(echo "$nvcc_version" | tr -d .); \
    else \
    echo "CUDA not detected, installing PyTorch for CPU use"; \
    pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu; \
    fi

# Adding environment variables
ENV interactivenet_results="/app/apps/interactivenet"

# Installing requirements
RUN pip install --no-cache-dir -r requirements.txt; \
    interactivenet_download_model -t $MODELS

# Adding environment variables <- these can be changed
ENV PORT=8080
ENV DATA=examples
ENV MODELS=all

# Exposing the port
EXPOSE $PORT
    
CMD monailabel start_server --app apps/interactivenet --studies $DATA --conf models $MODELS --port $PORT
