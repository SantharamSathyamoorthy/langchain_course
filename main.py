import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

print(".... initializing ....")

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
llm = ChatOllama(model="llama3.2:latest")

vectorstore = PineconeVectorStore(
    embedding=embeddings, index_name=os.getenv("INDEX_NAME")
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}
Question: {question}

Provide a detailed answer:"""
)


def format_docs(docs):
    """Format the retrieved documents into a string for the prompt."""
    return "\n\n".join([doc.page_content for doc in docs])


def retrival_chain_with_lcel():
    """Simple retrieval chain with LCEL.
    Returns a chain that can be invoked with {"question:"..."}
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


# without LCEL, we manually format the retrieved documents and generate a response.
def retrival_chain_without_lcel(query: str):
    """Simple retrieval chain without LCEL.
    Manually formats the retrieved documents and generates a response.
    """
    docs = retriever.invoke(query)
    context = format_docs(docs)

    messages = prompt.format_messages(context=context, question=query)

    response = llm.invoke(messages)
    return response.content


if __name__ == "__main__":
    print("Retrieving")

    # Query
    # query = "What is Pinecone in machine learning?"
    # response = retrival_chain_without_lcel(query)
    # print("Response: ", response)

    # Query with LCEL
    query = "What is Pinecone in machine learning?"
    response = retrival_chain_with_lcel().invoke({"question": query})
    print("Response: ", response)
