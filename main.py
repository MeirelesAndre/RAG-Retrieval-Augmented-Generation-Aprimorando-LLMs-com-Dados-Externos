import os
import sys
import pandas as pd
from dotenv import load_dotenv, find_dotenv, set_key
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredHTMLLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# 1. Load .env
print("⏳ Importando environment variables...")
load_dotenv(find_dotenv())
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 2. Configurações de seleção do arquivo .env
print("🔥 Verificando arquivos em ../data\n\n")


def selecionar_arquivo(diretorio="data"):
    arquivos = os.listdir(diretorio)
    arquivos = [f for f in arquivos if f.endswith(
        (".txt", ".pdf", ".md", ".csv", ".docx", ".html", ".htm"))]

    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum arquivo suportado encontrado em {diretorio}")

    print("\n📂 Selecione o arquivo desejado:\n")
    for i, nome in enumerate(arquivos):
        print(f"{i+1}. {nome}")

    print("\n💡 Digite 'sair' para cancelar a seleção.")

    while True:
        escolha = input(
            "\nDigite o número do arquivo desejado:\n> ").strip().lower()

        if escolha in ["sair", "exit", "cancelar"]:
            print("❌ Seleção de arquivo cancelada. Encerrando.")
            sys.exit(0)

        if escolha.isdigit() and 1 <= int(escolha) <= len(arquivos):
            return os.path.join(diretorio, arquivos[int(escolha) - 1])

        print("⚠️ Entrada inválida. Tente novamente.")


# Uso
ARQUIVO = os.getenv("ARQUIVO")
if not ARQUIVO:
    ARQUIVO = selecionar_arquivo()
    set_key(".env", "ARQUIVO", ARQUIVO)

print(f"\n📄 Arquivo selecionado: {ARQUIVO}")

# 3. Carrega documentos (detecta tipo)
extensao = ARQUIVO.lower().split(".")[-1]

if extensao == "txt" or extensao == "md":
    loader = TextLoader(ARQUIVO)
elif extensao == "pdf":
    loader = PyPDFLoader(ARQUIVO)
elif extensao == "docx":
    loader = UnstructuredWordDocumentLoader(ARQUIVO)
elif extensao in ["html", "htm"]:
    loader = UnstructuredHTMLLoader(ARQUIVO)
elif extensao == "csv":
    # Converte CSV em documentos
    df = pd.read_csv(ARQUIVO)
    documents = [Document(page_content=row.to_string(index=False))
                 for _, row in df.iterrows()]
    loader = None
else:
    raise ValueError(f"❌ Formato de arquivo não suportado: .{extensao}")

# Carrega os documentos
if loader:
    documents = loader.load()

# 4. Divide texto
print("🔪 dividindo o documento em pedaços menores...\n\n")
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
docs = text_splitter.split_documents(documents)
print(f"Total de pedaços: {len(docs)}\n\n")

# 5. Gera embeddings
print("⏳ Criando o cliente do Pinecone...")
embeddings = CohereEmbeddings(
    model="embed-english-v3.0", cohere_api_key=COHERE_API_KEY)

# 6. Cria vectorstore no Pinecone
INDEX_NAME = "rag-demo"
vectorstore_from_docs = PineconeVectorStore.from_documents(
    docs,
    index_name=INDEX_NAME,
    embedding=embeddings
)

# 7. Modelo LLM
print("⏳ Criando Modelo e RAG Chain...")
llm = ChatGroq(
    model="Gemma2-9b-It",
    groq_api_key=GROQ_API_KEY,
    temperature=0.1
)

# 8. Memória (opcional, se quiser manter contexto de conversa)
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# 9. Cria chain RAG
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt)
retriever = vectorstore_from_docs.as_retriever(search_kwargs={"k": 3})
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# 10. Função interativa


def fazer_pergunta_rag():
    while True:
        pergunta = input(
            "\n ❓ No que posso ajudar ❓ (ou digite 'sair' para encerrar):\n> ")

        if pergunta.lower() in ["sair", "exit", "quit"]:
            print("👋 Encerrando. Até logo!")
            break

        try:
            resposta = rag_chain.invoke(
                {"input": f"{pergunta} Responda em português"})
            print(f'\n🔍 Pergunta: {pergunta}')
            print("\n🧠 Resposta:\n")
            print(resposta["answer"])
        except Exception as e:
            print(f"\n⚠️ Erro ao gerar resposta: {e}")


# 11. Executa no final
if __name__ == "__main__":
    fazer_pergunta_rag()
