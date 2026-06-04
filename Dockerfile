# GATsig shear-jamming classifier
# Base: PyTorch with CUDA 12.1 (matches most vast.ai RTX/A100 instances)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    rsync \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create data and output directories
RUN mkdir -p /workspace/data /workspace/outputs

# Default: run training (override with docker run ... python src/train.py --args)
ENTRYPOINT ["python", "src/train.py"]
CMD ["--help"]
