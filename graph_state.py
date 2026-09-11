"""
State schema for the agentic RAG graph. Every node reads from and writes
to this shared state as the query moves through the pipeline.
"""

from typing import Literal, Optional, TypedDict


class AgentState(TypedDict):
    query: str
    route: Optional[Literal["sql", "vector", "both"]]
    sql_result: Optional[str]
    retrieved_chunks: Optional[list[str]]
    draft_answer: Optional[str]
    is_grounded: Optional[bool]
    retry_count: int
