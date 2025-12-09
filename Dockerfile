# Imagem Python leve
FROM python:3.11-slim

# --- Variáveis de Ambiente (Qwen 2.5 GGUF) ---
# Define o nome e a URL do modelo.
ENV GGUF_MODEL_NAME="qwen2.5-1.5b-instruct-q4_k_m.gguf"
ENV GGUF_MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
ENV FLASK_APP="src.api.main"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# --- 1. Instalação de Sistema ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libblas-dev \
    liblapack-dev \
    gcc \
    curl \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# --- 2. Dependências Python ---
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# --- 3. Código e Dados ---
COPY src/ /app/src/
COPY data/pdfs/ /app/data/pdfs/
RUN mkdir -p /app/data/vectorstore /app/data/models

# --- 4. Entrypoint ---
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]