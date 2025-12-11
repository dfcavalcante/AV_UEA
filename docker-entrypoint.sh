#!/bin/bash
set -e

# Ajuste este caminho dependendo de onde o COPY colocou o arquivo
# Se você copiou para /app/src/ingest.py ou /app/ingest.py
INGEST_SCRIPT="/app/src/ingest.py" 
MODEL_DIR="/app/data/models"
MODEL_PATH="${MODEL_DIR}/${GGUF_MODEL_NAME}"
VECTOR_STORE_PATH="/app/data/vectorstore"

echo "--- Iniciando Assistente Virtual UEA ---"

# --- ETAPA 1: Download do Modelo (Seguro) ---
if [ ! -f "$MODEL_PATH" ]; then
  echo "--- ⬇Modelo não encontrado. Baixando..."
  
  # Baixa para um arquivo temporário primeiro
  curl -L -f "$GGUF_MODEL_URL" -o "${MODEL_PATH}.tmp"
  
  if [ $? -eq 0 ]; then
    mv "${MODEL_PATH}.tmp" "$MODEL_PATH"
    echo "--- Download concluído! ---"
  else
    echo "--- Falha no download. ---"
    rm -f "${MODEL_PATH}.tmp"
    exit 1
  fi
else
  echo "--- Modelo já existe em cache. ---"
fi

# --- ETAPA 2: Ingestão de Dados ---
# Verifica se o script de ingestão existe antes de tentar rodar
if [ -f "$INGEST_SCRIPT" ]; then
    # Verifica se precisa rodar (se vetorstore está vazio)
    if [ ! -d "$VECTOR_STORE_PATH" ] || [ -z "$(ls -A $VECTOR_STORE_PATH)" ]; then
        echo "--- Criando Banco Vetorial... ---"
        python "$INGEST_SCRIPT"
    else
        echo "--- Banco Vetorial já existe. ---"
    fi
else
    echo "--- AVISO: Script $INGEST_SCRIPT não encontrado. Pulando ingestão. ---"
fi

# --- ETAPA 3: Iniciar Servidor ---
echo "--- Iniciando Uvicorn ---"
# --app-dir src é vital se seu código está dentro da pasta src/
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --app-dir src