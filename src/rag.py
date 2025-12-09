import os
import sys
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp

VECTOR_STORE_DIR = os.path.join("data", "vectorstore")

# --- Configurações Qwen 2.5 1.5B GGUF ---
GGUF_MODEL_NAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
GGUF_MODEL_PATH = os.path.join("data", "models", GGUF_MODEL_NAME)

LLAMA_N_GPU_LAYERS = 0   
LLAMA_N_CTX = 4096       
MAX_NEW_TOKENS = 1024    
TOP_K = 7 

# Prompt Template
PROMPT_TEMPLATE = """<|im_start|>system
Você é o Assistente Virtual da UEA. Responda com base no contexto fornecido.
Se a informação não estiver no contexto, diga APENAS: "A informação não consta nos documentos consultados."<|im_end|>
<|im_start|>user
Contexto:
{context}

Pergunta: {question}<|im_end|>
<|im_start|>assistant
"""

def initialize_llm():
    print(f"Carregando LLM (Qwen 2.5 GGUF)...", flush=True)
    try:
        llm = LlamaCpp(
            model_path=GGUF_MODEL_PATH,
            n_ctx=LLAMA_N_CTX,
            n_gpu_layers=LLAMA_N_GPU_LAYERS,
            temperature=0.1, 
            max_tokens=MAX_NEW_TOKENS,
            echo=False,
            repeat_penalty=1.15,
            stop=["<|im_end|>", "<|endoftext|>", "Contexto:", "Pergunta:"]
        )
        return llm 
    except Exception as e:
        print(f"Erro LLM: {e}", flush=True)
        return None

def initialize_vector_store():
    print(f"Carregando FAISS...", flush=True)
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
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
        
        # Recuperação
        results_with_scores = FAISS_INDEX.similarity_search_with_score(question, k=TOP_K)
        
        if not results_with_scores:
            return "A informação não consta nos documentos consultados."

        relevant_docs = []
        print("DEBUG SCORES (Sem Filtro):", flush=True)
        for i, (doc, score) in enumerate(results_with_scores):
            # Mostra o score para debug
            print(f"   [{i+1}] Score: {score:.4f} | Texto: {doc.page_content[:40].replace(chr(10), ' ')}...", flush=True)
            relevant_docs.append(doc)

        context_text = "\n---\n".join([doc.page_content for doc in relevant_docs])
        final_prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)

        print(f"Gerando resposta...", flush=True)
        answer = LLM_INSTANCE.invoke(final_prompt)
        
        return answer.strip()
        
    except Exception as e:
        print(f"ERRO: {e}", flush=True)
        return f"Erro na geração: {e}"

if __name__ == "__main__":
    print("Teste...")