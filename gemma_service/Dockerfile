FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8889

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8889"]
