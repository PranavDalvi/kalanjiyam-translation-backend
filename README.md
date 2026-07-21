# Kalanjiyam Translation API

FastAPI service built from your `translation.py` logic (HF IndicTrans2 models, GPU selection, offline local model loading, and DOCX/PDF/TXT document translation).

## What this provides

- `GET /health` for status and GPU visibility
- `GET /languages` for supported language names/codes
- `GET /models` for available translation model names and supported directions
- `POST /translate/text` for text translation
- `POST /translate/document` for document translation (`.docx`, `.pdf`, `.txt`) returning translated `.docx`
- `GET /glossaries` to list all available glossaries on the system

## Setup

```bash
cd /home/ganesh/kalanjiyam-translation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
cd /home/ganesh/kalanjiyam-translation
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8888 --reload
```

## Run Tests

Unit tests (fast, mocked translation path):

```bash
cd /home/ganesh/kalanjiyam-translation
source .venv/bin/activate
python -m pytest -q
```

Run a specific translation endpoint test only:

```bash
cd /home/ganesh/kalanjiyam-translation
source .venv/bin/activate
python -m pytest -q tests/test_health.py -k translate_text
```

Expected result for both commands: all selected tests should pass (for example `4 passed`).

## Docker Setup & Running

You can containerize the service to run with a modern CUDA toolkit independent of the host's CUDA toolkit version, or run it on a CPU-only system.

### Quick Start (Automated Script)

An automated setup and run script `setup_and_run.sh` is provided. This script will:
1. Build the Docker image.
2. Check if a GPU is available and whether Docker supports NVIDIA GPU reservations.
3. Automatically start the container in **GPU mode** (via Docker Compose) if available, or fall back to **CPU mode** (via Docker Run).
4. Run in online mode to download models on the first translation request if they are not already cached.

To use the script, run:
```bash
./setup_and_run.sh
```

### Manual Steps

#### 1. Build the Docker Image

Build the image locally:

```bash
docker build -t kalanjiyam-translation .
```

### 2. Run Tests in Docker

Run the unit tests inside the container (using dummy/mocked models):

```bash
docker run --rm kalanjiyam-translation python -m pytest -q
```

### 3. Run Backend API

#### Option A: Running with GPU (via Docker Compose)
Make sure the **NVIDIA Container Toolkit** is installed on the host. Run:

```bash
docker compose up -d
```

This will automatically:
- Bind port `8888` on the host.
- Expose all host GPUs to the container.
- Mount your host's Hugging Face cache `~/.cache/huggingface` inside the container (required for offline mode).

#### Option B: Running on CPU-only Systems
To run on a CPU-only machine, mount your local Hugging Face cache and start the container without GPU dependencies:

```bash
docker run -d \
  -p 8888:8888 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name kalanjiyam-translation-api \
  kalanjiyam-translation
```

### 4. Check Container Logs

To monitor the application logs:

- If running via **Docker Compose**:
  ```bash
  docker compose logs -f
  ```
- If running via **Docker Run**:
  ```bash
  docker logs -f kalanjiyam-translation-api
  ```

### 5. Downloading Models (First-Time Run or Empty Cache)

By default, the service expects models to be already cached locally (`TRANSFORMERS_OFFLINE=1`). If your host's Hugging Face cache (`~/.cache/huggingface`) is empty or missing the IndicTrans2 models, loading will fail.

> [!IMPORTANT]
> The `ai4bharat/indictrans2` models are **gated** on Hugging Face. To download them:
> 1. Log into your Hugging Face account and accept the terms of the model repository: [ai4bharat/indictrans2-en-indic-1B](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B).
> 2. Create a User Access Token (Read permission) under [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens).
> 3. Provide this token when prompted by `./setup_and_run.sh`, or pass it as the `HF_TOKEN` environment variable in the manual steps below.

To automatically download the models from Hugging Face on the first translation request and save them directly to your host's cache folder (via the volume mount):

- **Via Docker Run**:
  Pass `TRANSFORMERS_OFFLINE=0` and `HF_HUB_OFFLINE=0` environment variables:
  ```bash
  docker run -d \
    -p 8888:8888 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e TRANSFORMERS_OFFLINE=0 \
    -e HF_HUB_OFFLINE=0 \
    --name kalanjiyam-translation-api \
    kalanjiyam-translation
  ```

- **Via Docker Compose**:
  Pass the environment variables inline before starting compose:
  ```bash
  TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 docker compose up -d
  ```

*Note: Once the models are downloaded, future restarts of the container can be run in the default offline mode, as the models will persist in your host's `~/.cache/huggingface` directory.*



## Available Models

Use `GET /models` to fetch the translation models supported by this API.

Example response:

```json
[
  {
    "model_name": "ai4bharat/indictrans2-en-indic-1B",
    "key": "en-indic",
    "description": "English to Indic translation model",
    "source_languages": ["English"],
    "target_languages": ["Hindi", "Bengali", "Tamil"]
  }
]
```

## API Endpoints: Inputs and Expected Outputs

### `GET /health`

- Input: none
- Expected output (`200 OK`):

```json
{
  "status": "ok",
  "available_gpus": [0, 1, 2, 3],
  "offline_mode": {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1"
  }
}
```

### `GET /languages`

- Input: none
- Expected output (`200 OK`): JSON object with language name to language code mapping.

Example:

```json
{
  "English": "eng_Latn",
  "Hindi": "hin_Deva",
  "Tamil": "tam_Taml"
}
```

### `POST /translate/text`

- Input (`application/json`):

```json
{
  "text": "Hello world",
  "model_name": "ai4bharat/indictrans2-en-indic-1B",
  "source_language": "English",
  "target_language": "Hindi",
  "gpu_id": 0,
  "batch_size": 8,
  "glossary": "administrative"
}
```
*(Note: `glossary` is optional. Pass the glossary prefix or name, e.g. `"administrative"`, `"agriculture"`, etc.)*

- Expected output (`200 OK`):

```json
{
  "text": "...translated text..."
}
```

- Common error outputs:
  - `400`: invalid language names or invalid source/target combination
  - `400`: GPU id not available
  - `500`: no GPUs detected

### `POST /translate/document`

- Input (`multipart/form-data`):
  - `file`: `.docx`, `.pdf`, or `.txt`
  - `model_name`: one of the values returned by `GET /models`
  - `source_language`: e.g. `English`
  - `target_language`: e.g. `Hindi`
  - `gpu_id`: integer (default `0`)
  - `batch_size`: integer (default `8`)
  - `glossary`: string (optional, e.g. `"administrative"`)

- Expected output (`200 OK`): file download response (`.docx`) with translated content.

- Common error outputs:
  - `400`: unsupported file type
  - `400`: invalid language names
  - `400`: GPU id not available
  - `500`: no GPUs detected

### `GET /glossaries`

- Input: none
- Expected output (`200 OK`): List of available glossaries on the disk.

Example output:
```json
[
  {
    "name": "administrative",
    "source_language_code": "en",
    "target_language_code": "mr",
    "filename": "administrative_en_mr.csv"
  }
]
```

## Glossary Support

### 1. Configuration
Set the path to your glossary CSV files in your `.env` file:
```env
GLOSSARIES_DIR=/path/to/your/glossaries/directory
```
If using Docker, both direct execution (`setup_and_run.sh`) and Docker Compose dynamically mount this host directory to the container.

### 2. Glossary File Structure & Naming
- **Naming Pattern**: `[glossary_name]_[src_lang_code]_[tgt_lang_code].csv` (e.g. `administrative_en_mr.csv`).
- **CSV Format**:
  ```csv
  English word,Target translation
  ```
  Example:
  ```csv
  Wrongful dismissal,चुकीच्या पद्धतीने बाद करणे
  Yield,उत्पन्न
  ```

### 3. Glossary Name Aliases
You can request glossaries using their full names or common abbreviations. The API automatically maps them to short names on disk:
- `"agriculture"`, `"agriculture"` -> `agri`
- `"mechanical"`, `"mechanical"` -> `mech`
- `"biology"`, `"biology"` -> `bio`
- `"chemistry"`, `"chemistry"` -> `chem`
- `"computer"`, `"computer science"` -> `comp`
- `"physics"` -> `phy`
- `"mathematics"` -> `math`
- `"information technology"` -> `it`

### 4. Multi-Glossary Selection & "all" Option
You can request multiple glossaries or load all matching glossaries:
- **Formats Supported**:
  - **Single String**: `"administrative"`
  - **JSON List**: `["administrative", "agri"]`
  - **Comma-Separated String**: `"administrative, agri"`
- **"all" Option**: If you specify `"all"` (e.g. `"all"`, `["all"]`, or `"administrative, all"`), the API dynamically resolves it by loading and merging all available glossary files for the requested source/target language pair on the disk.
- **Precedence & Collision Resolution**:
  - Glossaries are merged in the order they are passed (later ones override earlier ones).
  - For `"all"`, files are loaded and merged in alphabetical order.

---

## Example calls

Health check:

```bash
curl -sS http://127.0.0.1:8000/health
```

Translate text:

```bash
curl -sS -X POST "http://127.0.0.1:8000/translate/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "model_name": "ai4bharat/indictrans2-en-indic-1B",
    "source_language": "English",
    "target_language": "Hindi",
    "gpu_id": 0,
    "batch_size": 8
  }'
```

Translate a document:

```bash
curl -X POST "http://127.0.0.1:8000/translate/document" \
  -F "file=@/absolute/path/input.docx" \
  -F "model_name=ai4bharat/indictrans2-en-indic-1B" \
  -F "source_language=English" \
  -F "target_language=Hindi" \
  -F "gpu_id=0" \
  -F "batch_size=8" \
  --output translated_output.docx
```

---

## API Key Authentication & Key Management

API key verification is supported to secure translation endpoints (`/translate/text` and `/translate/document`).

### 1. Configuration
Set the following environment variables in `.env`:
```env
# Enable API Key authentication (1 = enabled, 0 = disabled)
ENABLE_API_KEY_AUTH=1

# Path to SQLite database storing hashed API keys
API_KEY_DB_PATH=api_keys.db
```

### 2. CLI Key Management Tool
Use the CLI tool `app.manage_keys` to create, list, and revoke API keys:

#### Local Virtualenv Execution
* **Generate a new API key**:
  ```bash
  python -m app.manage_keys create --name "Mobile App"
  ```
* **List all API keys**:
  ```bash
  python -m app.manage_keys list
  ```
* **Revoke an API key**:
  ```bash
  python -m app.manage_keys revoke kt_a1b2
  # Or by ID:
  python -m app.manage_keys revoke 1
  ```

#### Docker & Docker Compose Execution
Run the CLI against a running container (or create a key in the volume-mounted database):

* **Via Docker / Podman (`docker exec` / `podman exec`)**:
  ```bash
  docker exec -it kalanjiyam-translation-api python -m app.manage_keys create --name "Mobile App"
  docker exec -it kalanjiyam-translation-api python -m app.manage_keys list
  docker exec -it kalanjiyam-translation-api python -m app.manage_keys revoke kt_a1b2
  ```
  *(For Podman, replace `docker` with `podman`)*

* **Via Docker Compose (`docker compose exec`)**:
  ```bash
  docker compose exec translation-api python -m app.manage_keys create --name "Mobile App"
  docker compose exec translation-api python -m app.manage_keys list
  docker compose exec translation-api python -m app.manage_keys revoke kt_a1b2
  ```


### 3. Using API Keys in Requests
When `ENABLE_API_KEY_AUTH=1`, pass your API key via the `X-API-Key` HTTP header:

```bash
curl -sS -X POST "http://127.0.0.1:8000/translate/text" \
  -H "X-API-Key: kt_a1b2c3d4e5f678901234567890abcdef" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "model_name": "ai4bharat/indictrans2-en-indic-1B",
    "source_language": "English",
    "target_language": "Hindi"
  }'
```

---

## Notes

- API keeps your original offline behavior (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).
- Models are loaded lazily on first request and cached per `(gpu_id, model_type)`.
- You must have local model files available in Hugging Face cache for offline mode.

