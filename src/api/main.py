from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import get_rag_answer, LLM_INSTANCE, FAISS_INDEX

app = FastAPI(
    title="Assistente Virtual UEA - RAG",
    description="API para responder perguntas sobre documentos institucionais da UEA usando TinyLlama GGUF (Quantizado) e FAISS."
)

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

@app.on_event("startup")
async def startup_event():
    if LLM_INSTANCE is None or FAISS_INDEX is None:
        print("CRITICAL: Falha na inicialização do LLM ou FAISS. Verifique logs.")

@app.post("/ask", response_model=Answer)
async def ask_question(body: Question):
    question = body.question.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="A pergunta não pode ser vazia.")
    
    response_text = get_rag_answer(question)
    
    if "Erro de inicialização" in response_text or "Erro durante a geração" in response_text:
        raise HTTPException(status_code=500, detail=response_text)
        
    return {"answer": response_text}