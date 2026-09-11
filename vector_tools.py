"""
Wraps a retriever around the FAQ vector store.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = Path("data/chroma")
COLLECTION_NAME = "banking_faq"
TOP_K = 3


def build_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


def query_faq(question: str) -> list[str]:
    retriever = build_retriever()
    docs = retriever.invoke(question)
    return [doc.page_content for doc in docs]


if __name__ == "__main__":
    test_question = "What is your policy on large transactions?"
    for chunk in query_faq(test_question):
        print(chunk)
        print("---")