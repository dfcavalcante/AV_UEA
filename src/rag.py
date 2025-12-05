import os
import sys
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp

VECTOR_STORE_DIR = os.path.join("data", "vectorstore")

# --- Configurações ---
GGUF_MODEL_NAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" 
GGUF_MODEL_PATH = os.path.join("data", "models", GGUF_MODEL_NAME)
LLAMA_N_GPU_LAYERS = 0   
LLAMA_N_CTX = 2048       
MAX_NEW_TOKENS = 256     
TOP_K = 5 
SCORE_THRESHOLD = 1.0 

PROMPT_TEMPLATE = """[INST] Use o contexto abaixo para responder. Se a resposta não estiver no texto, diga "Informação não encontrada".

Contexto:
{context}

Pergunta: {question} [/INST]
"""

def initialize_llm():
    print(f"Carregando LLM GGUF...", flush=True)
    try:
        llm = LlamaCpp(
            model_path=GGUF_MODEL_PATH,
            n_ctx=LLAMA_N_CTX,
            n_gpu_layers=LLAMA_N_GPU_LAYERS,
            temperature=0.1, 
            max_tokens=MAX_NEW_TOKENS,
            echo=False,
            repeat_penalty=1.1,
            stop=["[/INST]", "</s>", "Contexto:", "Pergunta:"]
        )
        return llm 
    except Exception as e:
        print(f"Erro LLM: {e}", flush=True)
        return None

def initialize_vector_store():
    print(f"Carregando FAISS...", flush=True)
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        print(f"Erro FAISS: {e}", flush=True)
        return None

LLM_INSTANCE = initialize_llm()
FAISS_INDEX = initialize_vector_store()

def get_rag_answer(question: str) -> str:
    if not LLM_INSTANCE or not FAISS_INDEX:
        return "Erro de inicialização."

    try:
        print(f"\n--- Pergunta: {question} ---", flush=True)
        
        # Busca com Score
        results_with_scores = FAISS_INDEX.similarity_search_with_score(question, k=TOP_K)
        
        if not results_with_scores:
            return "A informação não consta nos documentos consultados."

        # Filtra documentos
        relevant_docs = []
        print("DEBUG SCORES:", flush=True)
        for doc, score in results_with_scores:
            # Score menor = mais similar. 
            print(f"   - Score: {score:.4f} | Texto: {doc.page_content[:30]}...", flush=True)
            if score < SCORE_THRESHOLD:
                relevant_docs.append(doc)
        
        if not relevant_docs:
            print("   -> Bloqueado pelo filtro de relevância.", flush=True)
            return "A informação não consta nos documentos consultados."

        context_text = "\n---\n".join([doc.page_content for doc in relevant_docs])
        final_prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)

        print(f"Gerando resposta...", flush=True)
        answer = LLM_INSTANCE.invoke(final_prompt)
        
        # Limpeza
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[-1]
        
        answer = answer.strip()

        if "Use o contexto abaixo" in answer or "Se a resposta não estiver" in answer:
            return "A informação não consta nos documentos consultados."

        return answer
        
    except Exception as e:
        print(f"ERRO: {e}", flush=True)
        return f"Erro na geração: {e}"

if __name__ == "__main__":
    print("Teste...")