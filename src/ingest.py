import os
import glob
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Atualizado para evitar o warning
from langchain_community.vectorstores import FAISS

# --- CONFIGURAÇÃO DE CAMINHOS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
PDFS_PATH = os.path.join(DATA_PATH, 'pdfs')
VECTORSTORE_PATH = os.path.join(DATA_PATH, 'vectorstore')

def create_vector_db():
    print(f"--- Configuração de Caminhos ---")
    print(f"Raiz do Projeto: {PROJECT_ROOT}")
    print(f"Lendo PDFs de: {PDFS_PATH}")
    print(f"Salvando Vectorstore em: {VECTORSTORE_PATH}\n")

    # Verifica se a pasta de PDFs existe
    if not os.path.exists(PDFS_PATH):
        print(f"ERRO: A pasta {PDFS_PATH} não existe. Crie a pasta e coloque os PDFs lá.")
        return

    # Carregar PDFs
    print("--- 1. Carregando PDFs ---")
    loader = DirectoryLoader(PDFS_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("Nenhum documento encontrado.")
        return
        
    print(f"> {len(documents)} páginas carregadas.")

    # Dividir em Chunks
    print("--- 2. Dividindo em Chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    print(f"> {len(texts)} chunks gerados.")

    # Criar Vector Store
    print("--- 3. Gerando Embeddings e Indexando ---")
    
    # Modelo Embedder
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vectorstore = FAISS.from_documents(texts, embeddings)

    # Cria a pasta vectorstore se ela não existir
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)

    # Salva
    vectorstore.save_local(VECTORSTORE_PATH)
    
    print(f"\n--- SUCESSO! ---")
    print(f"Banco vetorial salvo corretamente em: {VECTORSTORE_PATH}")

if __name__ == "__main__":
    create_vector_db()