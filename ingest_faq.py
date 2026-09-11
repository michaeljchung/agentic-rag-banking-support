"""
Loads the FAQ CSV, embeds each Q&A pair, and writes it to a Chroma
vector store.
"""

from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

RAW_CSV_PATH = Path("data/raw/banking_faq.csv")
PERSIST_DIR = Path("data/chroma")
COLLECTION_NAME = "banking_faq"

QUESTION_COL = "Question"
ANSWER_COL = "Answer"


def build_documents(csv_path: Path) -> list[Document]:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    documents = []
    for i, row in df.iterrows():
        text = f"Q: {row[QUESTION_COL]}\nA: {row[ANSWER_COL]}"
        documents.append(
            Document(page_content=text, metadata={"source_row": i})
        )
    return documents


def ingest_faq(csv_path: Path, persist_dir: Path, collection_name: str) -> None:
    documents = build_documents(csv_path)
    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )

    print(f"Embedded and stored {len(documents)} FAQ chunks in {persist_dir}")


if __name__ == "__main__":
    ingest_faq(RAW_CSV_PATH, PERSIST_DIR, COLLECTION_NAME)
