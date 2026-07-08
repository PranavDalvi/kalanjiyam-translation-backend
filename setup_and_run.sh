#!/usr/bin/env bash
set -euo pipefail

echo "========================================================="
echo " Kalanjiyam Translation API - Setup & Run Script"
echo "========================================================="

# 1. Ensure Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed. Please install Docker first."
    exit 1
fi

# 2. Build the Docker Image
echo "Building the Docker image..."
docker build -t kalanjiyam-translation .

# 3. Check GPU Availability
echo "Checking GPU availability..."
HAS_GPU=false

if command -v nvidia-smi &> /dev/null; then
    # Test if nvidia-smi runs successfully and communicates with driver
    if nvidia-smi &> /dev/null; then
        # Check if docker runtime has GPU capabilities
        if docker run --help | grep -q "--gpus"; then
            HAS_GPU=true
        fi
    fi
fi

# 4. Ensure Hugging Face cache directory exists on host
mkdir -p ~/.cache/huggingface

# 5. Check if models are already cached locally
OFFLINE_MODE=1
CACHE_DIR="$HOME/.cache/huggingface/hub/models--ai4bharat--indictrans2-en-indic-1B"
HF_TOKEN_ENV="${HF_TOKEN:-}"

if [ ! -d "$CACHE_DIR" ]; then
    echo "---------------------------------------------------------"
    echo "Hugging Face model access authentication:"
    echo "No cached models found. Running in online mode to download them."
    echo "IndicTrans2 models are gated on Hugging Face. If they are not already cached"
    echo "locally, please accept the license terms at:"
    echo "  https://huggingface.co/ai4bharat/indictrans2-en-indic-1B"
    echo "and generate a read access token at https://huggingface.co/settings/tokens."
    echo "---------------------------------------------------------"
    if [ -z "$HF_TOKEN_ENV" ]; then
        read -rp "Enter your Hugging Face Access Token (press Enter to skip): " input_token
        HF_TOKEN_ENV="$input_token"
    fi
    OFFLINE_MODE=0
else
    echo "---------------------------------------------------------"
    echo "STATUS: Cached translation models detected."
    echo "Action: Running in strict offline mode (no token/network required)."
    echo "---------------------------------------------------------"
fi

# 6. Start Container
if [ "$HAS_GPU" = true ]; then
    echo "---------------------------------------------------------"
    echo "STATUS: NVIDIA GPU and Docker runtime detected."
    echo "Action: Starting container with GPU support using docker-compose..."
    echo "---------------------------------------------------------"
    
    # Try using docker compose (v2) or fallback to docker-compose (v1)
    if docker compose version &> /dev/null; then
        docker compose down --remove-orphans || true
        HF_TOKEN="$HF_TOKEN_ENV" TRANSFORMERS_OFFLINE="$OFFLINE_MODE" HF_HUB_OFFLINE="$OFFLINE_MODE" docker compose up -d
        echo "Service is running on http://localhost:8888"
        echo "To view logs, run: docker compose logs -f"
    elif command -v docker-compose &> /dev/null; then
        docker-compose down --remove-orphans || true
        HF_TOKEN="$HF_TOKEN_ENV" TRANSFORMERS_OFFLINE="$OFFLINE_MODE" HF_HUB_OFFLINE="$OFFLINE_MODE" docker-compose up -d
        echo "Service is running on http://localhost:8888"
        echo "To view logs, run: docker-compose logs -f"
    else
        echo "Warning: docker-compose command not found. Running with direct docker command..."
        docker rm -f kalanjiyam-translation-api 2>/dev/null || true
        docker run -d \
          -p 8888:8888 \
          --gpus all \
          -v ~/.cache/huggingface:/root/.cache/huggingface \
          -e TRANSFORMERS_OFFLINE="$OFFLINE_MODE" \
          -e HF_HUB_OFFLINE="$OFFLINE_MODE" \
          -e HF_TOKEN="$HF_TOKEN_ENV" \
          --name kalanjiyam-translation-api \
          kalanjiyam-translation
        echo "Service is running on http://localhost:8888"
        echo "To view logs, run: docker logs -f kalanjiyam-translation-api"
    fi
fi

if [ "$HAS_GPU" = false ]; then
    echo "---------------------------------------------------------"
    echo "STATUS: No GPU / NVIDIA Docker support detected."
    echo "Action: Starting container in CPU-only mode..."
    echo "---------------------------------------------------------"
    
    docker rm -f kalanjiyam-translation-api 2>/dev/null || true
    
    docker run -d \
      -p 8888:8888 \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      -e TRANSFORMERS_OFFLINE="$OFFLINE_MODE" \
      -e HF_HUB_OFFLINE="$OFFLINE_MODE" \
      -e HF_TOKEN="$HF_TOKEN_ENV" \
      --name kalanjiyam-translation-api \
      kalanjiyam-translation
      
    echo "Service is running on http://localhost:8888"
    echo "To view logs, run: docker logs -f kalanjiyam-translation-api"
fi

echo "========================================================="
echo "Note: The first translation request will download models"
echo "if they are not already cached. Please monitor the logs."
echo "========================================================="
