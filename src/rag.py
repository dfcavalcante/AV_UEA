import os
import sys
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp

VECTOR_STORE_DIR = os.path.join("data", "vectorstore")

# --- Configurações Gemma 2 2B ---
# Nome do arquivo que será baixado pelo Docker
GGUF_MODEL_NAME = "gemma-2-2b-it-Q4_K_M.gguf" 
GGUF_MODEL_PATH = os.path.join("data", "models", GGUF_MODEL_NAME)

LLAMA_N_GPU_LAYERS = 0   
LLAMA_N_CTX = 4096       
MAX_NEW_TOKENS = 1024    
TOP_K = 12               # Alta recuperação para garantir contexto completo

# --- Prompt Template (Padrão Oficial Gemma) ---
# O Gemma usa <start_of_turn> para separar turnos de conversa.
PROMPT_TEMPLATE = """<start_of_turn>user
Você é o Assistente Virtual da UEA. Responda APENAS com base no contexto fornecido abaixo.

Diretrizes:
1. Responda em Português do Brasil.
2. Seja direto e objetivo.
3. Se a informação não estiver no texto, diga APENAS: "Informação não encontrada".

Contexto:
{context}

Pergunta: {question}<end_of_turn>
<start_of_turn>model
"""

def initialize_llm():
    print(f"Carregando LLM (Gemma 2 2B)...", flush=True)
    try:
        llm = LlamaCpp(
            model_path=GGUF_MODEL_PATH,
            n_ctx=LLAMA_N_CTX,
            n_gpu_layers=LLAMA_N_GPU_LAYERS,
            temperature=0.1, # Baixa temperatura para precisão factual
            max_tokens=MAX_NEW_TOKENS,
            echo=False,
            repeat_penalty=1.15,
            # Tokens de parada específicos do Gemma
            stop=["<end_of_turn>", "user:", "model:"]
        )
        return llm 
    except Exception as e:
        print(f"Erro LLM: {e}", flush=True)
        return None

def initialize_vector_store():
    print(f"Carregando FAISS...", flush=True)
    try:
        # Mantemos o modelo Multilíngue (excelente para Português)
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
        relevant_docs = FAISS_INDEX.similarity_search(question, k=TOP_K)
        
        if not relevant_docs:
            return "A informação não consta nos documentos consultados."

        # Debug Visual para confirmar recuperação
        print(f"DEBUG: Analisando {len(relevant_docs)} trechos.", flush=True)
        found_key = False
        for i, doc in enumerate(relevant_docs):
            if "vedado" in doc.page_content.lower() or "proibido" in doc.page_content.lower():
                found_key = True
                print(f"   [ALVO] Trecho {i+1} contém termos de proibição.", flush=True)
        
        if not found_key:
             print("   [INFO] Palavras-chave exatas não encontradas nos tops (mas o contexto pode estar lá).", flush=True)

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