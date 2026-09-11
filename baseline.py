from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

from vector_tools import query_faq

load_dotenv()

LLM_MODEL = "claude-sonnet-4-5"
llm = ChatAnthropic(model=LLM_MODEL, temperature=0)


def naive_rag_answer(query: str) -> str:
    return naive_rag_full(query)["answer"]


def naive_rag_full(query: str) -> dict:
    """Same as naive_rag_answer, but also returns the retrieved chunks
    needed for eval scoring."""
    chunks = query_faq(query)
    context = "\n\n".join(chunks)

    prompt = (
        "Answer the user's question using ONLY the context below. "
        "If the context doesn't fully answer the question, say what's "
        "missing rather than guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    response = llm.invoke(prompt)
    return {"answer": response.content, "contexts": chunks}


if __name__ == "__main__":
    test_query = "What's my balance for AC00001, and what's your policy on balance holds?"
    print(naive_rag_answer(test_query))