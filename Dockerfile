# Imagem Base
FROM python:3.10-slim

# Configurações de Ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model_cache

# Instala dependências do sistema operacional
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Define diretório de trabalho
WORKDIR /app

# Copia e instala as dependências Python
COPY requirements.txt .
# Instala libs com --no-cache-dir para reduzir o tamanho da imagem final
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o código do projeto
COPY . .

# Prepara o volume e permissões
RUN mkdir -p /app/data/vectorstore && chmod 777 /app/data/vectorstore

# Converte quebras de linha Windows -> Linux e dá permissão
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

# Expõe a porta que o FastAPI usa
EXPOSE 8000

# Comando de Inicialização
CMD ["./start.sh"]