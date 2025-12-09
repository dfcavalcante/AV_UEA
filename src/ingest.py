import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configurações ---
PDF_URLS = {
    "estatuto_uea.pdf": "https://data.uea.edu.br/ssgp/area/1/est/442-1.pdf",
    "regimento_casas_estudante.pdf": "https://data.uea.edu.br/ssgp/legislacao/9563.pdf"
}

DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vectorstore")
CHUNK_SIZE = 500 
CHUNK_OVERLAP = 150 
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def download_pdfs(pdf_urls, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    print(f"\n--- 1. Baixando PDFs ---")
    for filename, url in pdf_urls.items():
        filepath = os.path.join(target_dir, filename)
        if not os.path.exists(filepath):
            print(f"  > Baixando {filename}...")
            try:
                response = requests.get(url, verify=False)
                response.raise_for_status()
                with open(filepath, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"  > ERRO: {e}")
        else:
            print(f"  > {filename} já existe.")

def load_documents(pdf_dir):
    documents = []
    print(f"\n--- 2. Carregando PDFs ---")
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
            print(f"  > {filename}: Carregado.")
    return documents

def split_documents(documents):
    print(f"\n--- 3. Dividindo em Chunks ({CHUNK_SIZE} chars) ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Separadores jurídicos para manter Artigos juntos
        separators=["\nArt. ", "\nI. ", "\nII. ", "\nIII. ", "\nIV. ", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  > Total de {len(chunks)} chunks gerados.")
    return chunks

def create_vector_store(chunks):
    print(f"\nIndexando no FAISS ({EMBEDDING_MODEL_NAME}) ---")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_DIR)
    print(f"Índice salvo em: {VECTOR_STORE_DIR}")

if __name__ == "__main__":
    download_pdfs(PDF_URLS, PDF_DIR)
    docs = load_documents(PDF_DIR)
    if docs:
        chunks = split_documents(docs)
        create_vector_store(chunks)