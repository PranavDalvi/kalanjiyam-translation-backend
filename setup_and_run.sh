#!/usr/bin/env bash
set -euo pipefail

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    while IFS= read -r line || [ -n "$line" ]; do
        # Strip carriage returns
        line=$(echo "$line" | tr -d '\r')
        # Skip empty lines and lines starting with '#'
        if [[ ! "$line" =~ ^# ]] && [[ ! -z "$line" ]]; then
            export "$line"
        fi
    done < .env
fi

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
        # Check if docker runtime has GPU capabilities or nvidia runtime registered
        if docker run --help 2>/dev/null | grep -q "--gpus" || docker info 2>/dev/null | grep -iq "nvidia"; then
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

# Clean HF_TOKEN_ENV by stripping leading/trailing whitespace, quotes, and any "token=" prefix
if [ -n "$HF_TOKEN_ENV" ]; then
    HF_TOKEN_ENV=$(echo "$HF_TOKEN_ENV" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^["'\''\\]*//' -e 's/["'\''\\]*$//' -e 's/^token=//')
fi

# Detect if the weights file exists, is not incomplete, and is larger than 10MB
CACHE_COMPLETE=0
if [ -d "$CACHE_DIR" ]; then
    # Follow symlinks (-L) to verify the actual downloaded blob files are >10MB
    if [ -n "$(find -L "$CACHE_DIR" -type f \( -name "model.safetensors" -o -name "pytorch_model.bin" \) -size +10M 2>/dev/null)" ]; then
        CACHE_COMPLETE=1
    fi
fi

if [ "$CACHE_COMPLETE" -eq 0 ] && [ -d "$CACHE_DIR" ]; then
    echo "---------------------------------------------------------"
    echo "WARNING: An incomplete or corrupted cache directory was detected at:"
    echo "  $CACHE_DIR"
    echo "To avoid download/resume issues, it is recommended to clean this directory."
    echo "---------------------------------------------------------"
    if [ -t 0 ]; then
        read -rp "Would you like to delete the incomplete cache and start fresh? (y/N): " clean_choice
        if [[ "$clean_choice" =~ ^[Yy]$ ]]; then
            echo "Cleaning cache directory..."
            rm -rf "$CACHE_DIR"
        fi
    else
        echo "Non-interactive shell detected. Retaining existing cache directory for resume attempt."
    fi
fi

if [ "$CACHE_COMPLETE" -eq 0 ]; then
    echo "---------------------------------------------------------"
    echo "Hugging Face model access authentication:"
    echo "No cached models found or cache is incomplete. Running in online mode to download them."
    echo "IndicTrans2 models are gated on Hugging Face. If they are not already cached"
    echo "locally, please accept the license terms at:"
    echo "  https://huggingface.co/ai4bharat/indictrans2-en-indic-1B"
    echo "and generate a read access token at https://huggingface.co/settings/tokens."
    echo "---------------------------------------------------------"
    if [ -z "$HF_TOKEN_ENV" ]; then
        read -rp "Enter your Hugging Face Access Token (press Enter to skip): " input_token
        # Clean input_token too
        input_token=$(echo "$input_token" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^["'\''\\]*//' -e 's/["'\''\\]*$//' -e 's/^token=//')
        HF_TOKEN_ENV="$input_token"
    fi
    OFFLINE_MODE=0
else
    echo "---------------------------------------------------------"
    echo "STATUS: Cached translation models detected."
    echo "Action: Running in strict offline mode (no token/network required)."
    echo "---------------------------------------------------------"
fi

# Ensure GLOSSARIES_DIR is resolved and exists on host
GLOSSARIES_DIR_VAL="${GLOSSARIES_DIR:-glossaries}"
GLOSSARIES_HOST_DIR="$GLOSSARIES_DIR_VAL"
if [[ ! "$GLOSSARIES_HOST_DIR" =~ ^/ ]]; then
    GLOSSARIES_HOST_DIR="$(pwd)/$GLOSSARIES_HOST_DIR"
fi
mkdir -p "$GLOSSARIES_HOST_DIR"

# 6. Start Container
if [ "$HAS_GPU" = true ]; then
    echo "---------------------------------------------------------"
    echo "STATUS: NVIDIA GPU and Docker runtime detected."
    echo "Action: Starting container with GPU support using docker-compose..."
    echo "---------------------------------------------------------"
    
    # Try using docker compose (v2) or fallback to docker-compose (v1)
    if docker compose version &> /dev/null; then
        docker rm -f kalanjiyam-translation-api 2>/dev/null || true
        docker compose down --remove-orphans || true
        GLOSSARIES_DIR="$GLOSSARIES_DIR_VAL" HF_TOKEN="$HF_TOKEN_ENV" TRANSFORMERS_OFFLINE="$OFFLINE_MODE" HF_HUB_OFFLINE="$OFFLINE_MODE" docker compose up -d
        echo "Service is running on http://localhost:8888"
        echo "To view logs, run: docker compose logs -f"
    elif command -v docker-compose &> /dev/null; then
        docker rm -f kalanjiyam-translation-api 2>/dev/null || true
        docker-compose down --remove-orphans || true
        GLOSSARIES_DIR="$GLOSSARIES_DIR_VAL" HF_TOKEN="$HF_TOKEN_ENV" TRANSFORMERS_OFFLINE="$OFFLINE_MODE" HF_HUB_OFFLINE="$OFFLINE_MODE" docker-compose up -d
        echo "Service is running on http://localhost:8888"
        echo "To view logs, run: docker-compose logs -f"
    else
        echo "Warning: docker-compose command not found. Running with direct docker command..."
        docker rm -f kalanjiyam-translation-api 2>/dev/null || true
        docker run -d \
          -p 8888:8888 \
          --gpus all \
          -v ~/.cache/huggingface:/root/.cache/huggingface \
          -v "$GLOSSARIES_HOST_DIR:/app/glossaries" \
          -e TRANSFORMERS_OFFLINE="$OFFLINE_MODE" \
          -e HF_HUB_OFFLINE="$OFFLINE_MODE" \
          -e HF_TOKEN="$HF_TOKEN_ENV" \
          -e GLOSSARIES_DIR="glossaries" \
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
      -v "$GLOSSARIES_HOST_DIR:/app/glossaries" \
      -e TRANSFORMERS_OFFLINE="$OFFLINE_MODE" \
      -e HF_HUB_OFFLINE="$OFFLINE_MODE" \
      -e HF_TOKEN="$HF_TOKEN_ENV" \
      -e GLOSSARIES_DIR="glossaries" \
      --name kalanjiyam-translation-api \
      kalanjiyam-translation
      
    echo "Service is running on http://localhost:8888"
    echo "To view logs, run: docker logs -f kalanjiyam-translation-api"
fi

echo "========================================================="
echo "Note: The first translation request will download models"
echo "if they are not already cached. Please monitor the logs."
echo "========================================================="

if [ "$OFFLINE_MODE" = 0 ]; then
    echo ""
    echo "---------------------------------------------------------"
    echo "Tailing container logs to monitor model downloads."
    echo "Press Ctrl+C to exit log viewer (the container will continue in the background)."
    echo "---------------------------------------------------------"
    docker logs -f kalanjiyam-translation-api
fi
