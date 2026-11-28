import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    filepath = Path("mediumblog1.txt")
    text = filepath.read_text(encoding="utf-8")
    documents = Document(page_content=text, metadata={"source": str(filepath)})

    document = [documents]

    print("splitting")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents=document)
    print(f"created {len(texts)} chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

    print("ingesting...")
    PineconeVectorStore.from_documents(
        documents=texts, embedding=embeddings, index_name=os.getenv("INDEX_NAME")
    )
