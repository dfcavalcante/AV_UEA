# 1. Imagem Base
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 2. Instalação do Sistema (Adicionei openblas explicitamente para garantir)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    liblapack-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- ESTRATÉGIA DE PERFORMANCE ---
# Define flags para compilar o llama-cpp-python com suporte a BLAS (muito mais rápido na CPU)
ENV CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_LAPACK=ON"
ENV FORCE_CMAKE=1

# 3. Instalação do PyTorch CPU
RUN pip install --no-cache-dir --default-timeout=3600 --retries 10 \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Instalação dos Requisitos
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=3600 --retries 10 -r requirements.txt

# 5. Variáveis de Ambiente
ENV GGUF_MODEL_NAME="gemma-2-2b-it-Q4_K_M.gguf"
# URL direta para download
ENV GGUF_MODEL_URL="https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"

# 6. Cópia do Código
COPY src/ /app/src/

COPY data/pdfs/ /app/data/pdfs/

# 7. Pastas
RUN mkdir -p /app/data/vectorstore /app/data/models

# 8. Entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]