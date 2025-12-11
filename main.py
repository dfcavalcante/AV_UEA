from src.rag import RAGPipeline

def main():
    try:
        # Inicializa o sistema
        app = RAGPipeline()
        
        print("\n --- Assistente UEA Pronto ---")
        
        while True:
            question = input("\nPergunte (ou 'sair'): ")
            if question.lower() in ["sair", "exit"]:
                break
            
            print(" Pensando...")
            resposta = app.get_answer(question)
            print(f" Resposta: {resposta}")
            
    except Exception as e:
        print(f"\n Erro crítico: {e}")

if __name__ == "__main__":
    main()