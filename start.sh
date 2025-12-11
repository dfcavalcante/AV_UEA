#!/bin/bash

# Define o caminho do índice FAISS que será persistido no volume
VECTOR_INDEX="/app/data/vectorstore/index.faiss"

echo "Inicializando Serviço com Docker Compose"

# Checa se o banco de vetores FAISS já existe no volume
if [ ! -f "$VECTOR_INDEX" ]; then
    echo "Banco vetorial não encontrado. Iniciando ingestão de documentos..."
    # Garante permissão para escrever no volume
    chmod -R 777 /app/data/vectorstore
    
    # Executa a ingestão, cria os arquivos FAISS e .pkl
    python src/ingest.py
    
    echo "Ingestão finalizada. Banco criado com sucesso."
else
    echo "Banco vetorial FAISS encontrado. Pulando etapa de ingestão."
fi

# Inicia o servidor FastAPI
echo "Iniciando servidor Uvicorn..."
echo "Para acessar, use: http://localhost:8000/docs"
exec uvicorn api.main:app --host 0.0.0.0 --port 8000