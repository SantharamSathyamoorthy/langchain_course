import os

from dotenv import load_dotenv

from pinecone import Pinecone,ServerlessSpec

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("INDEX_NAME")

if __name__ =="__main__":
    pc = Pinecone(api_key=api_key)

    if pc.has_index(index_name):
        print(f"Deleting index {index_name}...")
        pc.delete_index(index_name)

    print(f"Creating index {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        type="dense",
        serverless=ServerlessSpec(cloud="aws", region="us-east-1"),
    )