import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



# URL dos PDFs 
PDF_URLS = {
    "estatuto_uea.pdf": "https://data.uea.edu.br/ssgp/area/1/est/442-1.pdf",
    "regimento_casas_estudante.pdf": "https://data.uea.edu.br/ssgp/legislacao/9563.pdf"
}

# Diretórios
DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vectorstore")

# Configurações de Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Modelo de Embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Funções do Pipeline
def download_pdfs(pdf_urls, target_dir):
    # Baixa os PDFs se eles ainda não existirem localmente
    os.makedirs(target_dir, exist_ok=True)
    print(f"\nBaixando PDFs para '{target_dir}'")
    
    for filename, url in pdf_urls.items():
        filepath = os.path.join(target_dir, filename)
        if not os.path.exists(filepath):
            print(f"  > Baixando {filename}...")
            try:
                # verify=False para contornar o erro SSLError
                response = requests.get(url, verify=False) 
                response.raise_for_status()
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"  > {filename} baixado com sucesso.")
            except requests.exceptions.RequestException as e:
                print(f"  > ERRO ao baixar {filename}: {e}")
        else:
            print(f"  > {filename} já existe.")

def load_documents(pdf_dir):
    documents = []
    print(f"\nCarregando e Extraindo Texto dos PDFs")
    
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(filepath)
            # Estende a lista com os documentos carregados do PDF atual
            documents.extend(loader.load())
            print(f"  > '{filename}' carregado.")
            
    return documents

# Divide os documentos carregados em chunks menores
def split_documents(documents, chunk_size, chunk_overlap):
    print(f"\nDividindo em Chunks (Tamanho: {chunk_size}, Overlap: {chunk_overlap})")
    
    # Divisor para documentos semi-estruturados
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Define separadores para tentar manter o contexto coerente
        separators=["\n\n", "\n", " ", ""] 
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  > Total de {len(chunks)} chunks gerados.")
    return chunks

# Gera embeddings e salva o índice FAISS
def create_and_persist_vector_store(chunks, model_name, vector_store_dir):
    print(f"\nGerando Embeddings e Persistindo FAISS ---")
    print(f"  > Carregando modelo de embeddings: {model_name}...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("  > Gerando embeddings e criando índice FAISS")
    
    # Cria o índice FAISS a partir dos chunks e embeddings
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Salva o índice no disco
    os.makedirs(vector_store_dir, exist_ok=True)
    vector_store.save_local(vector_store_dir)
    
    print(f"\nÍndice FAISS salvo com sucesso em: {vector_store_dir}")


if __name__ == "__main__":
    
    # Criação de diretórios necessários
    os.makedirs(PDF_DIR, exist_ok=True)

    
    download_pdfs(PDF_URLS, PDF_DIR)
    documents = load_documents(PDF_DIR)
    
    if documents:
        # Chunking
        chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # Geração de Embeddings e Persistência FAISS
        create_and_persist_vector_store(chunks, EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR)
    
    print("\nPIPELINE DE INGESTÃO CONCLUÍDO")