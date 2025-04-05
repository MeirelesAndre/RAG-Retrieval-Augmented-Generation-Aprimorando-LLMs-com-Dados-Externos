# RAG - Retrieval Augmented Generation

Projeto de demonstração de um sistema RAG (Retrieval-Augmented Generation) com LangChain, utilizando documentos locais (.pdf, .txt, .csv, .docx, .html) como fonte de conhecimento para uma LLM.

> Repositório: [github.com/MeirelesAndre/RAG-Retrieval-Augmented-Generation-Aprimorando-LLMs-com-Dados-Externos](https://github.com/MeirelesAndre/RAG-Retrieval-Augmented-Generation-Aprimorando-LLMs-com-Dados-Externos)

---

## Índice

1. [Visão Geral](#visão-geral)
2. [Instalação e Execução](#instalação-e-execução)
3. [Blocos do código explicados](#blocos-do-código-explicados)
4. [Glossário](#glossário)
5. [Exemplo de uso](#exemplo-de-uso)

---

## Visão Geral

Esse projeto implementa um sistema RAG que:

- Permite ao usuário selecionar arquivos locais como fonte de conhecimento
- Utiliza LangChain + Groq + Cohere + Pinecone para processar e responder perguntas
- Suporta arquivos: `.txt`, `.pdf`, `.csv`, `.docx`, `.html`, `.md`

---

## Instalação e Execução

### 1. Clone o repositório:

```bash
git clone https://github.com/MeirelesAndre/RAG-Retrieval-Augmented-Generation-Aprimorando-LLMs-com-Dados-Externos
cd RAG-Retrieval-Augmented-Generation-Aprimorando-LLMs-com-Dados-Externos
```

### 2. Crie o ambiente virtual e instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
pip install -r requirements.txt
```

### 3. Configure suas chaves no arquivo `.env`:

```ini
COHERE_API_KEY=xxxxxx
GROQ_API_KEY=xxxxxx
PINECONE_API_KEY=xxxxxx
```

### 4. Execute o sistema:

```bash
python main.py
```

---

## Blocos do código explicados

### 1. Carregamento de variáveis de ambiente
Utiliza `dotenv` para carregar as chaves da API do arquivo `.env`.

### 2. Seleção de arquivo
Mostra ao usuário os arquivos suportados na pasta `data/` e permite a escolha.

### 3. Carregamento do documento
Detecta a extensão e aplica o loader correspondente (TextLoader, PyPDFLoader, etc).

### 4. Fragmentação de texto
Divide o conteúdo em partes menores para facilitar a indexação.

### 5. Gera embeddings
Usa `CohereEmbeddings` para transformar texto em vetores semânticos.

### 6. Cria vectorstore (Pinecone)
Indexa os documentos usando Pinecone para buscas vetoriais.

### 7. LLM (ChatGroq)
Define o modelo LLM que gerará respostas a partir dos documentos relevantes.

### 8. Memória de conversa (opcional)
Mantém contexto da conversa (pode ser expandido depois).

### 9. Cria a RAG chain
Combina busca com geração usando `create_retrieval_chain`.

### 10. Função interativa
Usuário digita perguntas e recebe respostas com base no documento selecionado.

---

## Glossário

| Termo           | Definição |
|----------------|------------|
| **RAG**        | Retrieval-Augmented Generation: geração de texto com apoio de busca externa. |
| **LLM**        | Large Language Model. Ex: ChatGPT, Gemma, LLaMA. |
| **Embedding**  | Vetor que representa o significado de um texto. |
| **Pinecone**   | Banco vetorial para busca por similaridade. |
| **LangChain**  | Framework para conectar LLMs com outras fontes e fluxos. |
| **Groq**       | Plataforma para executar LLMs com alta performance. |

---

## Exemplo de uso

```bash
python main.py

# Saída:
📂 Selecione o arquivo desejado:
1. exemplo.pdf
2. texto.txt
...

❓ No que posso ajudar ❓
> Qual é o assunto do documento?

🧠 Resposta:
O documento aborda...
```

---


Feito com 💡 por [@MeirelesAndre](https://github.com/MeirelesAndre)

