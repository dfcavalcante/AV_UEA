import os
import fitz 
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import sys
import re 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# Remove cabeçalhos repetitivos e quebras de linha excessivas
def clean_text(text):
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove cabeçalhos comuns que sujam a busca
        if "UNIVERSIDADE DO ESTADO DO AMAZONAS" in line:
            continue
        if len(line.strip()) < 5: # Pula linhas muito curtas
            continue
        cleaned_lines.append(line)
    
    # Junta tudo de novo
    return " ".join(cleaned_lines)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return clean_text(text) # Limpeza aplicada

def chunk_text(text, source_name):
    chunks = []
    start = 0
    chunk_size = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP

    while start < len(text):
        end = start + chunk_size
        chunk_content = text[start:end]
        
        chunks.append({
            "text": chunk_content,
            "source": source_name
        })
        start = end - overlap
    return chunks

def ingest_data():
    print(" Iniciando ingestão com LIMPEZA DE TEXTO...")
    
    all_chunks_data = [] 
    texts_only = []      

    if not os.path.exists(config.PDF_DIR):
        print("Pasta data/pdfs não encontrada.")
        return

    files = [f for f in os.listdir(config.PDF_DIR) if f.endswith(".pdf")]
    
    for file_name in files:
        file_path = os.path.join(config.PDF_DIR, file_name)
        print(f"    Lendo e Limpando: {file_name}")

        full_text = extract_text_from_pdf(file_path)
        file_chunks = chunk_text(full_text, file_name)
        
        all_chunks_data.extend(file_chunks)
        texts_only.extend([c["text"] for c in file_chunks])
        
        print(f"      -> Gerou {len(file_chunks)} chunks.")

    print(f"Gerando embeddings para {len(texts_only)} trechos...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_ID)
    embeddings = model.encode(texts_only)
    embeddings = np.array(embeddings).astype("float32")

    print("Salvando banco de dados...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(config.VECTORSTORE_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(config.VECTORSTORE_PATH, "index.faiss"))
    
    with open(os.path.join(config.VECTORSTORE_PATH, "chunks.pkl"), "wb") as f:
        pickle.dump(all_chunks_data, f)

    print("Ingestão Concluída!")

if __name__ == "__main__":
    ingest_data()