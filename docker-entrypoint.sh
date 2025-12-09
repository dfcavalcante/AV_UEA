#!/bin/bash
# o script parar imediatamente se qualquer comando der erro
set -e

# --- Definição de Caminhos ---
MODEL_PATH="/app/data/models/${GGUF_MODEL_NAME}"
VECTOR_STORE_PATH="/app/data/vectorstore"
INGEST_SCRIPT="/app/src/ingest.py"

echo "--- Iniciando Assistente Virtual UEA (Engine: LlamaCpp / Qwen GGUF) ---"

# --- Download do Modelo LLM ---
if [ ! -f "$MODEL_PATH" ]; then
  echo "--- Modelo GGUF não encontrado. Iniciando download... ---"
  echo "URL: $GGUF_MODEL_URL"
  
  # Usa curl com -L para seguir redirecionamentos do Hugging Face
  curl -L "$GGUF_MODEL_URL" -o "$MODEL_PATH"
  
  if [ $? -eq 0 ]; then
    echo "--- Download GGUF concluído com sucesso. ---"
  else
    echo "--- Erro ao baixar o modelo. ---"
    exit 1
  fi
else
  echo "--- Modelo GGUF encontrado em cache. Pulando download. ---"
fi

# --- Ingestão de Dados (FAISS) ---
# Verifica se a pasta existe E se tem arquivos dentro
if [ ! -d "$VECTOR_STORE_PATH" ] || [ ! "$(ls -A $VECTOR_STORE_PATH)" ]; then
  echo "--- Índice Vetorial (FAISS) não encontrado ou vazio. ---"
  echo "--- Executando pipeline de ingestão (ingest.py)... ---"
  
  # Roda o script que baixa PDFs, faz chunking e salva os vetores
  python $INGEST_SCRIPT
  
  echo "--- Ingestão e Indexação FAISS concluídas! ---"
else
  echo "--- Índice Vetorial FAISS encontrado. Pulando etapa de ingestão. ---"
fi

# --- Iniciar a API ---
echo "--- Iniciando o servidor FastAPI (Uvicorn) na porta 8000... ---"

# Executa o Uvicorn.
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --app-dir src