#!/usr/bin/env bash
set -euo pipefail

curl -sS -X POST "http://127.0.0.1:8000/translate/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "source_language": "English",
    "target_language": "Hindi",
    "gpu_id": 0,
    "batch_size": 8
  }'
