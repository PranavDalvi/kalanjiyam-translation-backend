# Use an official PyTorch runtime base image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface

# Set working directory
WORKDIR /app

# Install git and build-essential (required for compiling IndicTransToolkit)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and mock files
COPY mock_modeling_indictrans.py .
COPY app/ ./app/
COPY tests/ ./tests/

# Expose the API port
EXPOSE 8888

# Run command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8888"]
