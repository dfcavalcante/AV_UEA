import os
import torch

# --- CAMINHOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
VECTORSTORE_PATH = os.path.join(BASE_DIR, "data", "vectorstore")

# --- MODELO ---
LLM_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cpu"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200 
RETRIEVAL_K = 10

# --- GERAÇÃO ---
GEN_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.1,
    "do_sample": True
}