# Imagem Python leve e estável
FROM python:3.11-slim

# Variáveis de Ambiente para o Modelo GGUF
ENV GGUF_MODEL_NAME "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
# URL para o download do arquivo GGUF 4-bit 
ENV GGUF_MODEL_URL "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
ENV FLASK_APP "src.api.main"
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Instalação das Ferramentas de Build e Dependências
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

# Copia requisitos e instala as dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia do Código e Estrutura de Dados
COPY src/ /app/src/
# Copia os PDFs fornecidos para o container (se já estiverem na pasta do host)
COPY data/pdfs/ /app/data/pdfs/
# Cria os diretórios necessários
RUN mkdir -p /app/data/vectorstore /app/data/models

# Configuração do Entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]