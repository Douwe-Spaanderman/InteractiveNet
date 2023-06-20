# syntax=docker/dockerfile:1
ARG MODELS="all"
ARG PORT=8080
ARG DATA="examples"

# Installing neccessary packages
FROM python:3.9-slim as base

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

# Installing requirements
RUN pip install --no-cache-dir -r requirements.txt; \
    export interactivenet_results="/app/apps/interactivenet/model"; \
    download_and_install_model -t $MODELS
    
CMD ["monailabel", "start_server", "--app", "apps/interactivenet", "--studies", "$DATA", "--conf", "models", "$MODELS", "--port", "$PORT"]
