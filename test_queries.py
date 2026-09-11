"""
Test queries for evaluating the agentic RAG pipeline.

Categories:
  faq_verbatim     - matches FAQ dataset phrasing closely (sanity check)
  faq_paraphrased  - reworded, tests real retrieval robustness
  sql_only         - needs a specific account fact, no policy needed
  sql_and_faq      - needs both an account fact AND a policy explanation
"""

TEST_QUERIES = [
    # --- faq_verbatim ---
    {"id": "q1", "category": "faq_verbatim", "query": "What is an alert transaction?"},
    {"id": "q2", "category": "faq_verbatim", "query": "What is a daily transaction limit?"},
    {"id": "q3", "category": "faq_verbatim", "query": "What is a bulk deposit?"},

    # --- faq_paraphrased ---
    {"id": "q4", "category": "faq_paraphrased", "query": "How much money can I move out of my account in one day?"},
    {"id": "q5", "category": "faq_paraphrased", "query": "If I deposit a really large amount at once, does anything special happen?"},
    {"id": "q6", "category": "faq_paraphrased", "query": "What kinds of transactions get flagged for review?"},

    # --- sql_only ---
    {"id": "q7", "category": "sql_only", "query": "What is the account balance for AC00001?"},
    {"id": "q8", "category": "sql_only", "query": "How many login attempts were recorded for account AC00019?"},
    {"id": "q9", "category": "sql_only", "query": "What device was used for account AC00128's most recent transaction?"},
    {"id": "q10", "category": "sql_only", "query": "What was the transaction amount and location for account AC00455?"},

    # --- sql_and_faq ---
    {
        "id": "q11",
        "category": "sql_and_faq",
        "query": "What's my balance for AC00001, and what's your policy on balance holds?",
    },
    {
        "id": "q12",
        "category": "sql_and_faq",
        "query": (
            "Was there any suspicious activity on account AC00001, and what "
            "should I do if I think my account was compromised?"
        ),
    },
    {
        "id": "q13",
        "category": "sql_and_faq",
        "query": (
            "Account AC00019 made a transaction of $126.29 — is that "
            "considered a large transaction, and what happens if it is?"
        ),
    },
]


if __name__ == "__main__":
    for item in TEST_QUERIES:
        print(f"[{item['category']}] {item['id']}: {item['query']}")