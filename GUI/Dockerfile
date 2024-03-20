# syntax=docker/dockerfile:1

# Installing neccessary packages
FROM python:3.9 as base

WORKDIR /app
RUN git clone https://github.com/Douwe-Spaanderman/InteractiveNet && cd /app/InteractiveNet && git checkout development

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
RUN pip install numpy
RUN cd /app/InteractiveNet/GUI && pip install -r requirements.txt
RUN pip install dicomweb-client[gcp]
ENV interactivenet_results="/app/InteractiveNet/GUI/apps/interactivenet/model"
RUN interactivenet_download_model -t all
    
# Optional arguments
ENV models="all"
ENV port="8080"
ENV studies="/app/InteractiveNet/GUI/examples"

CMD monailabel start_server --app /app/InteractiveNet/GUI/apps/interactivenet --studies $studies --conf models $models --port $port --host 0.0.0.0
