import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pincone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("INDEX_NAME")

if __name__ == "__main__":
    pc = Pinecone(api_key=pincone_api_key)

    if pc.has_index(index_name):
        print(f"index , {index_name} found, deleting...")
        pc.delete_index(index_name)

    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    print(f"index {index_name} is created")
