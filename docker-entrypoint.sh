#!/bin/bash
# 'set -e' faz o script parar imediatamente se qualquer comando der erro (segurança)
set -e

# --- Definição de Caminhos (Baseados nas Variáveis de Ambiente do Dockerfile) ---
# O caminho onde o modelo será salvo
MODEL_PATH="/app/data/models/${GGUF_MODEL_NAME}"
# O diretório do banco vetorial
VECTOR_STORE_PATH="/app/data/vectorstore"
# O script Python de ingestão
INGEST_SCRIPT="/app/src/ingest.py"

echo "--- 🐳 Iniciando Assistente Virtual UEA (Engine: LlamaCpp / GGUF) ---"

# --- ETAPA 1: Download do Modelo LLM ---
if [ ! -f "$MODEL_PATH" ]; then
  echo "--- ⬇️ Modelo GGUF não encontrado. Iniciando download... ---"
  echo "URL: $GGUF_MODEL_URL"
  
  # Usa curl com -L para seguir redirecionamentos do Hugging Face
  # O -f garante que falhe se o HTTP code for erro (404, 500)
  curl -L -f "$GGUF_MODEL_URL" -o "$MODEL_PATH"
  
  if [ $? -eq 0 ]; then
    echo "--- ✅ Download GGUF concluído com sucesso! ---"
  else
    echo "--- ❌ Erro ao baixar o modelo. Verifique sua conexão ou a URL. ---"
    exit 1
  fi
else
  echo "--- ✅ Modelo GGUF encontrado em cache. Pulando download. ---"
fi

# --- ETAPA 2: Ingestão de Dados (FAISS) ---
# Verifica se a pasta existe E se tem arquivos dentro
if [ ! -d "$VECTOR_STORE_PATH" ] || [ ! "$(ls -A $VECTOR_STORE_PATH)" ]; then
  echo "--- 🔄 Índice Vetorial (FAISS) não encontrado ou vazio. ---"
  echo "--- ▶️ Executando pipeline de ingestão (ingest.py)... ---"
  
  # Roda o script que baixa PDFs, faz chunking e salva os vetores
  python $INGEST_SCRIPT
  
  echo "--- ✅ Ingestão e Indexação FAISS concluídas! ---"
else
  echo "--- ✅ Índice Vetorial FAISS encontrado. Pulando etapa de ingestão. ---"
fi

# --- ETAPA 3: Iniciar a API (Servidor) ---
echo "--- 🚀 Iniciando o servidor FastAPI (Uvicorn) na porta 8000... ---"

# Executa o Uvicorn.
# --app-dir src: Define a pasta 'src' como a raiz para resolver as importações corretamente
# exec: Substitui o processo shell pelo Python, garantindo que o container pare corretamente
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --app-dir src