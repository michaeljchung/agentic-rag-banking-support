"""
Runs every test query through both the naive baseline and the full
agentic pipeline, scores both with RAGAS, and writes a comparison CSV.
"""

import time

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

from baseline import naive_rag_full
from graph import run_agent_full
from test_queries import TEST_QUERIES

OUTPUT_CSV = "eval_results.csv"


def collect_results() -> pd.DataFrame:
    rows = []

    for item in TEST_QUERIES:
        query = item["query"]
        print(f"Running: {item['id']} ({item['category']})")

        start = time.time()
        baseline_result = naive_rag_full(query)
        baseline_latency = time.time() - start

        start = time.time()
        pipeline_result = run_agent_full(query)
        pipeline_latency = time.time() - start

        pipeline_contexts = []
        if pipeline_result.get("sql_result"):
            pipeline_contexts.append(pipeline_result["sql_result"])
        if pipeline_result.get("retrieved_chunks"):
            pipeline_contexts.extend(pipeline_result["retrieved_chunks"])

        rows.append({
            "id": item["id"],
            "category": item["category"],
            "query": query,
            "baseline_answer": baseline_result["answer"],
            "baseline_contexts": baseline_result["contexts"],
            "baseline_latency_sec": round(baseline_latency, 2),
            "pipeline_answer": pipeline_result["draft_answer"],
            "pipeline_contexts": pipeline_contexts,
            "pipeline_latency_sec": round(pipeline_latency, 2),
            "route": pipeline_result.get("route"),
            "retry_count": pipeline_result.get("retry_count"),
        })

    return pd.DataFrame(rows)


def score_with_ragas(df: pd.DataFrame, answer_col: str, context_col: str) -> pd.DataFrame:
    dataset = Dataset.from_dict({
        "question": df["query"].tolist(),
        "answer": df[answer_col].tolist(),
        "contexts": df[context_col].tolist(),
    })
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    return result.to_pandas()


def main():
    df = collect_results()

    print("\nScoring baseline with RAGAS...")
    baseline_scores = score_with_ragas(df, "baseline_answer", "baseline_contexts")

    print("Scoring full pipeline with RAGAS...")
    pipeline_scores = score_with_ragas(df, "pipeline_answer", "pipeline_contexts")

    df["baseline_faithfulness"] = baseline_scores["faithfulness"]
    df["baseline_answer_relevancy"] = baseline_scores["answer_relevancy"]
    df["pipeline_faithfulness"] = pipeline_scores["faithfulness"]
    df["pipeline_answer_relevancy"] = pipeline_scores["answer_relevancy"]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved full results to {OUTPUT_CSV}")

    print("\n--- Summary ---")
    print(f"Baseline avg faithfulness:      {df['baseline_faithfulness'].mean():.2f}")
    print(f"Pipeline avg faithfulness:      {df['pipeline_faithfulness'].mean():.2f}")
    print(f"Baseline avg answer_relevancy:  {df['baseline_answer_relevancy'].mean():.2f}")
    print(f"Pipeline avg answer_relevancy:  {df['pipeline_answer_relevancy'].mean():.2f}")
    print(f"Baseline avg latency (sec):     {df['baseline_latency_sec'].mean():.2f}")
    print(f"Pipeline avg latency (sec):     {df['pipeline_latency_sec'].mean():.2f}")
    print(f"Reflection retry trigger rate:  {(df['retry_count'] > 1).mean():.0%}")


if __name__ == "__main__":
    main()