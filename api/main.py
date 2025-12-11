from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import sys
import os

# --- Configuração de Caminhos ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from src.rag import RAGPipeline

# --- Modelos de Dados ---
# Define o que a API aceita receber
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

# --- Inicialização da API ---
app = FastAPI(
    title="Assistente Virtual UEA API",
    description="API para interação com o modelo RAG local da UEA",
    version="1.0"
)

# Redireciona a raiz para a documentação
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# Variável global para o motor
rag_engine = None

# Carrega o modelo na memória assim que o servidor liga
@app.on_event("startup")
def load_model():
    global rag_engine
    print("Inicializando o motor RAG...")
    try:
        rag_engine = RAGPipeline()
        print("Modelo carregado e pronto!")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")

# --- 4. Rotas ---

# Verifica se a API e o Modelo estão online
@app.get("/health")
def health_check():
    status = "online" if rag_engine else "loading_or_error"
    return {"status": status, "device": "cpu"}

# Recebe uma pergunta e retorna a resposta do RAG.
@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    global rag_engine
    
    if not rag_engine:
        raise HTTPException(status_code=503, detail="O modelo ainda está carregando ou falhou. Tente novamente em instantes.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia.")

    print(f"📩 [FastAPI] Recebida: {request.question}")

    try:
        # Chama o motor RAG
        response_text = rag_engine.get_answer(request.question)
        
        return AnswerResponse(
            question=request.question,
            answer=response_text
        )
    except Exception as e:
        print(f"Erro interno: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. Executor (Para rodar via python api/main.py) ---
if __name__ == "__main__":
    import uvicorn
    # reload=False porque carregar o modelo demora, e reload recarrega a cada save
    uvicorn.run(app, host="0.0.0.0", port=8000)