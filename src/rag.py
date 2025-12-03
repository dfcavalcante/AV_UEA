import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from typing import List

VECTOR_STORE_DIR = os.path.join("data", "vectorstore")

# Configurações do LLM GGUF
GGUF_MODEL_NAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" 
GGUF_MODEL_PATH = os.path.join("data", "models", GGUF_MODEL_NAME) # Onde o modelo será baixado

# Configurações de Geração
LLAMA_N_GPU_LAYERS = 0   # Força o uso exclusivo da CPU
LLAMA_N_CTX = 2048       # Aumenta a janela de contexto para caber o prompt RAG
MAX_NEW_TOKENS = 256     # Limite de tokens de saída
TOP_K = 2                # Número de chunks

# Template de Prompt (Instrução)

PROMPT_TEMPLATE = """
[INST] Você é o Assistente Virtual UEA. Use APENAS o contexto fornecido abaixo para listar os requisitos de forma direta e concisa.
Não invente informações ou crie listas. Liste somente o que estiver no contexto.
Se a informação não estiver no contexto, responda: "A informação não foi encontrada nos documentos da UEA."

Contexto:
{context}

Pergunta: {question} [/INST]
"""

# Funções de Inicialização (Carregar LLM e FAISS)

def initialize_llm():
    """Inicializa o LLM local usando LlamaCpp e o modelo GGUF."""
    print(f"Carregando LLM GGUF (TinyLlama 1.1B) de {GGUF_MODEL_PATH}...")
    try:
        llm = LlamaCpp(
            model_path=GGUF_MODEL_PATH,
            n_ctx=LLAMA_N_CTX,
            n_gpu_layers=LLAMA_N_GPU_LAYERS,
            temperature=0.01, # Baixa temperatura para respostas factuais RAG
            max_tokens=MAX_NEW_TOKENS,
            verbose=False,
        )
        print("LLM GGUF carregado com sucesso.")
        return llm 
    except Exception as e:
        print(f"Erro ao carregar o LLM GGUF. Verifique o caminho GGUF. Erro: {e}")
        return None

# Carrega o índice FAISS persistido
def initialize_vector_store():
    print(f"Carregando índice FAISS de {VECTOR_STORE_DIR}...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Carrega o índice FAISS do disco
        vector_store = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
        print("Índice FAISS carregado com sucesso.")
        return vector_store.as_retriever(search_kwargs={"k": TOP_K})
    except Exception as e:
        print(f"Erro ao carregar o índice FAISS. Execute 'python src/ingest.py'. Erro: {e}")
        return None

# Variáveis globais para armazenar os componentes carregados
LLM_INSTANCE = initialize_llm()
FAISS_RETRIEVER = initialize_vector_store()

# Executa o pipeline RAG completo
def get_rag_answer(question: str) -> str:
    if not LLM_INSTANCE or not FAISS_RETRIEVER:
        return "Erro de inicialização do assistente. Verifique os logs do LLM ou FAISS."

    # Busca os K chunks mais relevantes no índice FAISS
    retrieved_docs = FAISS_RETRIEVER.invoke(question)
    
    # Converte os documentos em uma única string de contexto
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])

    # Monta o prompt final com o contexto recuperado e a pergunta do usuário
    final_prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # Geração
    print(f"Buscando resposta para: {question[:50]}...")
    
    # Chama o pipeline do LLM com parâmetros de sampling
    try:
        answer = LLM_INSTANCE.invoke(final_prompt)
        
    except Exception as e:
        return f"Erro durante a geração da resposta pelo LLM: {e}"

# --- Teste Simples (Opcional) ---
if __name__ == "__main__":
    # Teste para garantir que o LLM e FAISS carregam
    print("Teste de carga concluído. Tentando uma pergunta de teste...")
    test_question = "Quais são os requisitos que o aluno deve atender para concorrer a uma vaga nas Casas do Estudantes da UEA?"
    
    answer = get_rag_answer(test_question)
    
    print("\n" + "="*50)
    print(f"PERGUNTA: {test_question}")
    print(f"RESPOSTA RAG:")
    print(answer)
    print("="*50)