# 🤖 Assistente Virtual UEA - RAG Local

## 📄 Resumo sobre o projeto
Este projeto implementa um sistema de **RAG (Retrieval-Augmented Generation)** projetado para responder perguntas sobre documentos institucionais da **Universidade do Estado do Amazonas (UEA)**, especificamente o Estatuto e o Regimento das Casas do Estudante.

O diferencial deste projeto é sua capacidade de operar **100% localmente em CPU**, utilizando modelos de linguagem eficientes (Qwen-0.5B) e orquestração via Docker, reprodutibilidade e baixo consumo de recursos.

---

## 🏗️ Arquitetura Geral
O sistema segue uma arquitetura modular desacoplada em três camadas principais:

1.  **Camada de Ingestão (`src/ingest.py`):**
    * Responsável por ler os arquivos PDF da pasta `data/pdfs/`.
    * Utiliza **PyMuPDF** para extração limpa de texto.
    * Aplica técnica de *Chunking* (tamanho 1000, overlap 200) para preservar o contexto de artigos de lei.
    * Gera vetores (embeddings) e os armazena em um índice **FAISS**.

2.  **Camada RAG Core (`src/rag.py`):**
    * Atua como o motor de inteligência.
    * Realiza a busca vetorial para recuperar os trechos mais relevantes.
    * Implementa um **Re-ranking Híbrido** (detalhado nas funcionalidades adicionais) para refinar os resultados antes de enviá-los ao LLM.
    * Utiliza o modelo **Qwen/Qwen2.5-0.5B-Instruct** para gerar a resposta final em linguagem natural.

3.  **Camada de Interface (`api/main.py`):**
    * Servidor **FastAPI** que expõe as funcionalidades via HTTP.
    * Gerencia o ciclo de vida dos modelos, carregando-os apenas uma vez na inicialização.

---

## 🚀 Como rodar o projeto
O projeto utiliza **Docker Compose** para orquestração. Não é necessário instalar Python ou bibliotecas localmente, apenas o Docker.

1.  **Clone o repositório e entre na pasta:**
    ```bash
    git clone https://github.com/dfcavalcante/AV_UEA.git
    cd AV_UEA
    ```

2.  **Execute o comando de inicialização:**
    Este comando irá construir a imagem, baixar os modelos de IA e iniciar o serviço.
    ```bash
    docker-compose up --build
    ```
    *(Aguarde até aparecer a mensagem "✅ Servidor Online!" no terminal).*
    
    ** E acesse:**
    http://localhost:8000/docs

## 🐍 Execução Manual (Sem Docker - Opcional)
Caso prefira rodar o projeto diretamente em seu ambiente Python local (Windows/Linux/Mac).

**Pré-requisitos:** Python 3.10 ou superior.

1.  **Crie e ative um ambiente virtual:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Linux/Mac
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Instale as dependências:**
    O `requirements.txt` já está otimizado para baixar a versão leve (CPU) do PyTorch.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Realize a Ingestão dos Documentos:**
    Como não está usando o script automático do Docker, você deve criar o banco vetorial manualmente na primeira vez.
    ```bash
    python src/ingest.py
    ```

4.  **Inicie a API:**
    ```bash
    uvicorn api.main:app --host localhost --port 8000 --reload
    ```

5.  **Acesse:**
    http://localhost:8000/docs

---

## 🔄 Como regenerar o índice
O banco de dados vetorial (FAISS) é salvo em um volume do Docker para evitar reprocessamento desnecessário. Caso você adicione novos PDFs ou altere as configurações de chunking, é necessário forçar a recriação do índice:

1.  Pare o serviço e **remova os volumes** (isso apaga o banco atual):
    ```bash
    docker-compose down -v
    ```
2.  Suba a aplicação novamente:
    ```bash
    docker-compose up --build
    ```
    *O sistema detectará automaticamente que o índice não existe e executará o script de ingestão antes de iniciar a API.*

### Via Execução Manual (Sem Docker)
Se você estiver rodando o projeto diretamente no Python localmente:

1.  **Apague a pasta do banco antigo:**
    Delete manualmente a pasta `data/vectorstore` (ou os arquivos `index.faiss` e `chunks.pkl` dentro dela).
    * **Windows (PowerShell):** `Remove-Item -Recurse -Force data/vectorstore`
    * **Linux/Mac:** `rm -rf data/vectorstore`

2.  **Rode o script de ingestão:**
    ```bash
    python src/ingest.py
    ```

---

## 🔌 Como chamar a API

1.  Com o projeto rodando, acesse **http://localhost:8000/docs** no seu navegador.
2.  Localize a rota **`POST /ask`** e clique nela.
3.  Clique no botão **"Try it out"**.
4.  No campo **Request body**, envie sua pergunta no formato JSON:
    ```json
    {
      "question": "Segundo o Estatuto, como é constituído o patrimônio da Universidade?"
    }
    ```
5.  Clique em **"Execute"** e veja a resposta no campo **Response body**.

---

## ✅ Funcionalidades obrigatórias implementadas
O projeto atende a 100% dos requisitos solicitados no desafio:

1.  **Pipeline de Ingestão de Documentos:** Leitura de PDFs, chunking e indexação vetorial automática.
2.  **Execução de Modelo LLM Local:** Integração com modelo *Open Source* rodando localmente.
3.  **Otimização para CPU:** Configuração explícita de `torch --index-url .../cpu` e uso de modelos leves para execução sem GPU.
4.  **Pipeline de RAG:** Implementação completa do fluxo de Recuperação (Retrieval) e Geração (Generation).
5.  **API HTTP:** Criação de endpoint REST via FastAPI.
6.  **Dockerização:** Criação de `Dockerfile` e `docker-compose.yml` funcionais.

---

## ✨ Funcionalidades adicionais implementadas
Para garantir maior qualidade e robustez, foram implementadas as seguintes funcionalidades extras:

1.  **Busca Híbrida com Re-ranking:**
    * Além da busca vetorial simples, o sistema aplica um algoritmo de reclassificação. Ele pontua mais alto trechos que contêm palavras-chave exatas da pergunta e prioriza a fonte correta (ex: se a pergunta menciona "Estatuto", documentos com "Estatuto" no nome ganham prioridade), reduzindo alucinações.

2.  **Orquestração Inteligente (`start.sh`):**
    * Script shell personalizado que gerencia a lógica de inicialização. Ele verifica a existência do banco vetorial e decide automaticamente se deve rodar a ingestão ou iniciar a API diretamente, economizando tempo.

3.  **Tratamento de Compatibilidade (Windows/Linux):**
    * Configuração de `.dockerignore` e tratamento de quebras de linha (LF/CRLF) no Dockerfile para garantir que o projeto rode em qualquer sistema operacional sem erros de script.

4.  **Endpoint de Health Check (`GET /health`):**
    * Implementação de uma rota de monitoramento que retorna o status da aplicação.

## ⚠️ Observações Técnicas e Limitações

### Comportamento do Modelo (Small Language Model)
Este projeto utiliza o modelo **Qwen-0.5B**, uma versão extremamente leve projetada para rodar em CPUs modestas. Devido ao tamanho reduzido de parâmetros:
1.  **Alucinações de Conhecimento Externo:** Embora o *prompt* instrua o modelo a responder apenas sobre o contexto, modelos dessa escala (0.5B) podem ocasionalmente priorizar seu conhecimento prévio de treinamento em perguntas de conhecimento geral.
2.  **Decisão de Design:** Optou-se por **não implementar filtros rígidos** para a identificação de perguntas desconexas ao contexto fornecido. Testes mostraram que filtros rígidos tendem a gerar **Falsos Negativos**, bloqueando perguntas válidas sobre a universidade que utilizam termos comuns.
