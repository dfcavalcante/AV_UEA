import faiss
import pickle
import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class RAGPipeline:
    def __init__(self):
        print(f"  Carregando RAG na {config.DEVICE}...")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_ID)

        try:
            self.index = faiss.read_index(os.path.join(config.VECTORSTORE_PATH, "index.faiss"))
            with open(os.path.join(config.VECTORSTORE_PATH, "chunks.pkl"), "rb") as f:
                self.chunks_data = pickle.load(f)
        except:
            raise FileNotFoundError("Erro: Rode o 'python src/ingest.py' primeiro!")

        print(f" Carregando LLM: {config.LLM_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_ID,
            device_map=config.DEVICE,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.llm = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config.GEN_CONFIG["max_new_tokens"],
            temperature=0.1,
            do_sample=True,
            return_full_text=False
        )

    def _keyword_score(self, text, query):
        """Dá pontos extras se as palavras da query existirem no texto"""
        score = 0
        text_lower = text.lower()
        stopwords = ["o", "a", "os", "as", "de", "do", "da", "que", "é", "em", "para", "qual", "como"]
        keywords = [k for k in query.lower().split() if k not in stopwords and len(k) > 3]
        
        for word in keywords:
            if word in text_lower:
                score += 10 
        return score

    def get_answer(self, query):
        # 1. BUSCA AMPLA (Deep Retrieval)
        k_search = 100
        query_vec = self.embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vec, k_search)
        
        candidates = []
        for idx in indices[0]:
            if idx < len(self.chunks_data):
                candidates.append(self.chunks_data[idx])

        # 2. RE-RANKING HÍBRIDO
        ranked_candidates = []
        for c in candidates:
            priority_score = 0
            query_lower = query.lower()
            source = c['source'].lower()

            # Prioridade de Fonte
            if "estatuto" in query_lower and "estatuto" in source:
                priority_score += 1000
            elif ("regimento" in query_lower or "casa" in query_lower) and "regimento" in source:
                priority_score += 1000
            
            # Prioridade de Palavra-chave
            keyword_points = self._keyword_score(c['text'], query)
            
            total_score = priority_score + keyword_points
            ranked_candidates.append((total_score, c))

        ranked_candidates.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [item[1] for item in ranked_candidates[:config.RETRIEVAL_K]]

        # 3. CONTEXTO E DEBUG (Limpo)
        print(f"\n Contexto Selecionado ({len(top_chunks)} trechos):")
        context_text = ""
        for c in top_chunks:
            # Debug limpo apenas com nome do arquivo e inicio do texto
            clean_preview = c['text'][:60].replace('\n', ' ')
            print(f"   [{c['source']}]: {clean_preview}...")
            
            context_text += f"FONTE: {c['source']}\nCONTEÚDO: {c['text']}\n---\n"

        # 4. GERAÇÃO
        prompt = f"""<|im_start|>system
Você é um assistente da UEA. Responda à pergunta usando APENAS o contexto abaixo.
Cite os itens listados no texto fielmente.
Se não souber, diga "Não consta no texto". Responda em Português.<|im_end|>
<|im_start|>user
Contexto:
{context_text}

Pergunta:
{query}<|im_end|>
<|im_start|>assistant
"""
        outputs = self.llm(prompt)
        return outputs[0]["generated_text"].strip()