# Agentic RAG for Banking Support

Most RAG demos answer questions from a single document store and stop there. Real banking support queries don't work that way — a customer asking "what's my balance, and can I get a hold lifted early?" needs both a specific account fact and a policy explanation in the same answer. I built this project to explore that gap: an agentic system that routes between structured and unstructured data sources, and checks its own output before returning it, rather than a single-shot RAG pipeline that can only ever search one index.

## Problem

Real support queries often need two different kinds of information at once — a specific account fact (e.g. "what's my balance") and a general policy explanation (e.g. "what's your hold policy"). A standard single-source RAG pipeline can only search one index, so it structurally cannot answer questions that need both.

## Architecture

Router --> SQL branch and/or Vector branch (parallel) --> Synthesis --> Reflection
|
(loop back to Router,
bounded by MAX_RETRIES=2)
- **Router** — classifies each query as needing the SQL branch, vector branch, or both.
- **SQL branch** — a text-to-SQL agent that queries a transactions database for account-specific facts.
- **Vector branch** — similarity search over a Chroma-embedded FAQ knowledge base for policy questions.
- **Synthesis** — combines whatever context is available into one grounded answer. On a retry, it also receives feedback from the reflection step about what was wrong with the previous attempt.
- **Reflection** — checks whether the draft answer is actually grounded in the retrieved context. If not, it sends the query back through the router with feedback, bounded to 2 retries so it can't loop forever.

## Data Sources

- **Banking FAQ Dataset for NLP, RAG & Chatbot Dev** — [Kaggle](https://www.kaggle.com/datasets/rudrakumargupta/banking-faq-dataset-for-chatbot-training) (Apache 2.0)
- **Bank Transactions Dataset for Fraud Detection** — [Kaggle](https://www.kaggle.com/datasets/thuandao/bank-transactions-dataset-for-fraud-detection) (MIT)

Both datasets are used for portfolio/demonstration purposes and are not included in this repository — download them from the links above and place them in `data/raw/` before running the ingestion scripts. Note the FAQ dataset is India-centric (references ₹ and cybercrime.gov.in), which shapes some example answers.

## Stack

- **LLM**: Claude Sonnet (`claude-sonnet-4-5`) via `langchain_anthropic`, used for routing, synthesis, reflection, and the SQL agent
- **Orchestration**: LangGraph (state machine, conditional edges, parallel branch fan-out)
- **Vector store**: Chroma (`langchain_chroma`)
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Structured data**: SQLite
- **Evaluation**: RAGAS (faithfulness, answer relevancy)

## Evaluation

To measure whether the agentic architecture is actually worth its added complexity, the pipeline was benchmarked against a naive single-source RAG baseline (vector search only, no routing, no SQL access, no reflection) on the same 13 test queries spanning FAQ-verbatim, FAQ-paraphrased, SQL-only, and combined SQL+FAQ question types.

| Metric | Baseline | Full Pipeline |
|---|---|---|
| Faithfulness | 0.91 | 0.97 |
| Answer relevancy | 0.22 | 0.60 |
| Avg latency (sec) | 3.47 | 15.71 |

**Faithfulness** measures whether the answer is actually supported by the retrieved context, rather than hallucinated. Both approaches score well here, since neither is inclined to invent facts.

**Answer relevancy** measures whether the answer actually addresses the question asked — and this is the key result. The baseline correctly reports "I don't have that information" whenever asked about account-specific data, since it has no database access; that response is honest but unhelpful, and scores low on relevancy. The full pipeline's SQL branch lets it actually answer those questions, nearly tripling the relevancy score.

**Latency** is the tradeoff: routing, dual-branch retrieval, and reflection add real overhead versus a single vector lookup.

On this test set, every query passed reflection on the first attempt (0% retry rate) — the reflection/feedback loop functioned as a safety net that wasn't needed here rather than one that fired often. Its correct-behavior-on-failure was separately verified with an adversarial manual test.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with:

ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

Download the two datasets above into `data/raw/`, then run ingestion:
```bash
python ingest_transactions.py
python ingest_faq.py
```

Run the agent:
```bash
python graph.py
```

Run the evaluation harness:
```bash
python eval.py
```

## Project Structure
├── graph.py # LangGraph state machine (router, branches, synthesis, reflection)
├── graph_state.py # AgentState TypedDict
├── sql_tools.py # Text-to-SQL agent for transaction queries
├── vector_tools.py # Chroma retriever for FAQ queries
├── ingest_transactions.py # Loads transaction CSV into SQLite
├── ingest_faq.py # Loads FAQ CSV into Chroma
├── test_queries.py # 13 test queries across 4 categories
├── baseline.py # Naive single-source RAG for comparison
├── eval.py # Runs both pipelines through RAGAS, outputs eval_results.csv
└── requirements.txt
