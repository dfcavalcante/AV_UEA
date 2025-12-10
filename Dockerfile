# 1. Imagem Base
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 2. Instalação do Sistema
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

# --- ESTRATÉGIA ANTI-QUEDA DE CONEXÃO ---

# 3. Instalação do PyTorch (O arquivo de 1GB) SEPARADAMENTE
# Usamos --retries 10 para ele tentar de novo se a internet cair.
# Usamos o index-url da CPU para garantir a versão certa.
RUN pip install --no-cache-dir --default-timeout=3600 --retries 10 \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Instalação do Resto dos Requisitos
COPY requirements.txt .
# O pip vai ver que o torch já está instalado e vai pular ele (ficando rápido)
RUN pip install --no-cache-dir --default-timeout=3600 --retries 10 -r requirements.txt

# -----------------------------------------------------------

# 5. Configuração do Modelo (Gemma 2 2B GGUF)
ENV GGUF_MODEL_NAME="gemma-2-2b-it-Q4_K_M.gguf"
ENV GGUF_MODEL_URL="https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"
ENV FLASK_APP="src.api.main"

# 6. Cópia do Código Fonte e Dados
COPY src/ /app/src/
COPY data/pdfs/ /app/data/pdfs/

# 7. Estrutura de Pastas
RUN mkdir -p /app/data/vectorstore /app/data/models

# 8. Entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]