#!/bin/bash
# O 'set -e' garante que o script irá parar se qualquer comando falhar
set -e

# Define Caminhos e Variáveis
MODEL_PATH="/app/data/models/${GGUF_MODEL_NAME}"
VECTOR_STORE_PATH="/app/data/vectorstore"
INGEST_SCRIPT="/app/src/ingest.py"

echo "--- Iniciando Preparação de Dados para o Assistente Virtual UEA ---"

# Baixa Modelo GGUF
if [ ! -f "$MODEL_PATH" ]; then
  echo "--- Modelo GGUF não encontrado. Iniciando download... ---"
  echo "URL: $GGUF_MODEL_URL"
  curl -L "$GGUF_MODEL_URL" -o "$MODEL_PATH"
  echo "--- Download GGUF concluído. ---"
else
  echo "--- Modelo GGUF encontrado. Pulando download. ---"
fi

# Ingestão (Geração do Índice FAISS)
if [ ! -d "$VECTOR_STORE_PATH" ] || [ ! "$(ls -A $VECTOR_STORE_PATH)" ]; then
  echo "--- Índice Vetorial FAISS não encontrado. Iniciando pipeline de ingestão... ---"
  python $INGEST_SCRIPT
  echo "--- Ingestão e Indexação FAISS concluídas. ---"
else
  echo "--- Índice Vetorial FAISS encontrado. Pulando ingestão. ---"
fi

echo "--- INFRAESTRUTURA DE DADOS PRONTA. ---"

# Inicia a API 
echo "--- Iniciando o servidor FastAPI Uvicorn ---"
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --app-dir src