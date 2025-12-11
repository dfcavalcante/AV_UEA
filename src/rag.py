import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp

# --- Configuração de Caminhos ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) 
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vectorstore")
MODELS_DIR = os.path.join(DATA_DIR, "models")

# --- Configurações do Modelo ---
GGUF_MODEL_NAME = "gemma-2-2b-it-Q4_K_M.gguf"
GGUF_MODEL_PATH = os.path.join(MODELS_DIR, GGUF_MODEL_NAME)

# Parâmetros de Geração
LLAMA_N_GPU_LAYERS = 0   # 0 para CPU, aumentar se tiver GPU
LLAMA_N_CTX = 4096       # Janela de contexto do Gemma
MAX_NEW_TOKENS = 1024    
TOP_K = 10               

# --- Prompt Template ---
PROMPT_TEMPLATE = """<start_of_turn>user
Você é o Assistente Virtual da UEA. Responda APENAS com base no contexto fornecido abaixo.

Diretrizes:
1. Responda em Português do Brasil de forma clara.
2. Se a informação não estiver no texto, diga: "A informação não consta nos documentos consultados."
3. Cite o artigo ou parágrafo se estiver explícito no texto.

Contexto:
{context}

Pergunta: {question}<end_of_turn>
<start_of_turn>model
"""

# --- Variáveis Globais ---
LLM_INSTANCE = None
FAISS_INDEX = None

def initialize_resources():
    """Inicializa LLM e FAISS apenas uma vez."""
    global LLM_INSTANCE, FAISS_INDEX
    
    print("--- Inicializando Recursos RAG ---", flush=True)

    # 1. Carregar LLM
    if not LLM_INSTANCE:
        if os.path.exists(GGUF_MODEL_PATH):
            print(f"Carregando LLM: {GGUF_MODEL_NAME}...", flush=True)
            try:
                LLM_INSTANCE = LlamaCpp(
                    model_path=GGUF_MODEL_PATH,
                    n_ctx=LLAMA_N_CTX,
                    n_gpu_layers=LLAMA_N_GPU_LAYERS,
                    temperature=0.2, # Baixa temperatura para ser mais fiel ao texto
                    max_tokens=MAX_NEW_TOKENS,
                    echo=False,
                    stop=["<end_of_turn>", "user:", "model:"]
                )
            except Exception as e:
                print(f"ERRO FATAL ao carregar LLM: {e}", flush=True)
        else:
            print(f"ERRO: Modelo não encontrado em {GGUF_MODEL_PATH}", flush=True)

    # 2. Carregar Banco Vetorial (FAISS)
    if not FAISS_INDEX:
        if os.path.exists(VECTOR_STORE_DIR):
            print("Carregando índice FAISS...", flush=True)
            try:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                FAISS_INDEX = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"ERRO FATAL ao carregar FAISS: {e}", flush=True)
        else:
            print(f"ERRO: Pasta vectorstore não encontrada em {VECTOR_STORE_DIR}", flush=True)

# Inicializa na importação (ou chame explicitamente no startup da API)
initialize_resources()

def get_rag_answer(question: str) -> str:
    if not LLM_INSTANCE or not FAISS_INDEX:
        return "Erro interno: O sistema de IA não foi inicializado corretamente (Modelo ou Banco de Dados ausentes)."

    try:
        print(f"\n{'='*40}", flush=True)
        print(f"PERGUNTA: {question}", flush=True)
        
        # --- LÓGICA DE FILTRO (ROUTING) ---
        search_kwargs = {"k": TOP_K}
        question_lower = question.lower()
        
        # IMPORTANTE: Os valores de 'source' devem bater EXATAMENTE com o que está no ingest.py
        # ingest.py: doc.metadata["source"] = f"data/pdfs/{filename}"
        
        filtro_aplicado = "NENHUM (Busca Geral)"
        
        if "estatuto" in question_lower:
            search_kwargs["filter"] = {"source": "data/pdfs/estatuto_uea.pdf"}
            filtro_aplicado = "ESTATUTO UEA"
            
        elif any(termo in question_lower for termo in ["regimento", "casa do estudante", "moradia"]):
            search_kwargs["filter"] = {"source": "data/pdfs/regimento_casas_estudante.pdf"}
            filtro_aplicado = "REGIMENTO CASAS"

        print(f"DEBUG: Filtro de Contexto -> {filtro_aplicado}", flush=True)
        
        # Recuperação
        if "filter" in search_kwargs:
            relevant_docs = FAISS_INDEX.similarity_search(question, **search_kwargs)
        else:
            relevant_docs = FAISS_INDEX.similarity_search(question, k=TOP_K)
        
        if not relevant_docs:
            return "A informação não consta nos documentos consultados."

        # Debug dos trechos recuperados
        print(f"DEBUG: {len(relevant_docs)} trechos recuperados.", flush=True)
        

        # Montagem do Contexto
        context_text = "\n---\n".join([doc.page_content for doc in relevant_docs])
        
        # Geração
        final_prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)
        print(f"Gerando resposta...", flush=True)
        
        answer = LLM_INSTANCE.invoke(final_prompt)
        return answer.strip()
        
    except Exception as e:
        print(f"ERRO NA GERAÇÃO: {e}", flush=True)
        return "Desculpe, ocorreu um erro ao processar sua pergunta."