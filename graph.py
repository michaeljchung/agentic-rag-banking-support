"""
Assembles the agentic RAG pipeline as a LangGraph state machine:

    Router --> SQL branch and/or Vector branch --> Synthesis --> Reflection
                                                                     |
                                                      (loop back to Router,
                                                       bounded by MAX_RETRIES)

Quick manual test:
    python graph.py
"""

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv

from graph_state import AgentState
from sql_tools import query_transactions
from vector_tools import query_faq

load_dotenv()

MAX_RETRIES = 2
LLM_MODEL = "claude-sonnet-4-5"

llm = ChatAnthropic(model=LLM_MODEL, temperature=0)


# ---------- Nodes ----------

def router_node(state: AgentState) -> dict:
    """Classifies the query as needing the SQL branch, vector branch, or both."""
    prompt = (
        "You are routing a banking support query to the right data source.\n"
        "- 'sql': the query asks for a specific account fact (balance, "
        "login attempts, transaction details, device used).\n"
        "- 'vector': the query asks about general policy or how something "
        "works.\n"
        "- 'both': the query needs a specific account fact AND a policy "
        "explanation together.\n\n"
        f"Query: {state['query']}\n\n"
        "Respond with exactly one word: sql, vector, or both."
    )
    response = llm.invoke(prompt)
    route = response.content.strip().lower()
    if route not in ("sql", "vector", "both"):
        route = "both"  # safe fallback if the model doesn't follow format
    return {"route": route}


def sql_branch_node(state: AgentState) -> dict:
    result = query_transactions(state["query"])
    return {"sql_result": result}


def vector_branch_node(state: AgentState) -> dict:
    chunks = query_faq(state["query"])
    return {"retrieved_chunks": chunks}


def synthesis_node(state: AgentState) -> dict:
    context_parts = []
    if state.get("sql_result"):
        context_parts.append(f"Account data:\n{state['sql_result']}")
    if state.get("retrieved_chunks"):
        joined_chunks = "\n\n".join(state["retrieved_chunks"])
        context_parts.append(f"Policy/FAQ context:\n{joined_chunks}")
    context = "\n\n".join(context_parts)

    feedback_note = ""
    if state.get("feedback"):
        feedback_note = (
            "\n\nYour previous attempt had a problem: "
            f"{state['feedback']}\nFix this in your new answer."
        )

    prompt = (
        "Answer the user's question using ONLY the context below. "
        "If the context doesn't fully answer the question, say what's "
        "missing rather than guessing."
        f"{feedback_note}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}\n\n"
        "Answer:"
    )
    response = llm.invoke(prompt)
    return {"draft_answer": response.content}


def reflection_node(state: AgentState) -> dict:
    context_parts = []
    if state.get("sql_result"):
        context_parts.append(f"Account data:\n{state['sql_result']}")
    if state.get("retrieved_chunks"):
        joined_chunks = "\n\n".join(state["retrieved_chunks"])
        context_parts.append(f"Policy/FAQ context:\n{joined_chunks}")
    context = "\n\n".join(context_parts)

    prompt = (
        "Check whether the draft answer below is fully grounded in the "
        "provided context and actually answers the question.\n"
        "Respond in exactly this format:\n"
        "VERDICT: YES or NO\n"
        "REASON: one short sentence explaining why (only if NO)\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}\n"
        f"Draft answer: {state['draft_answer']}"
    )
    response = llm.invoke(prompt)
    text = response.content.strip()

    verdict = "YES" in text.split("\n")[0].upper()
    reason = None
    if not verdict:
        for line in text.split("\n"):
            if line.upper().startswith("REASON"):
                reason = line.split(":", 1)[-1].strip()
                break

    print(f"[reflection] grounded={verdict} reason={reason}")

    return {
        "is_grounded": verdict,
        "feedback": reason,
        "retry_count": state["retry_count"] + 1,
    }


# ---------- Conditional edges ----------

def route_decision(state: AgentState):
    route = state["route"]
    if route == "sql":
        return ["sql_branch"]
    if route == "vector":
        return ["vector_branch"]
    return ["sql_branch", "vector_branch"]  # "both"


def reflection_decision(state: AgentState) -> str:
    if state["is_grounded"]:
        return END
    if state["retry_count"] >= MAX_RETRIES:
        return END  # give up, return best-effort draft
    return "router"


# ---------- Graph assembly ----------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("sql_branch", sql_branch_node)
    graph.add_node("vector_branch", vector_branch_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("reflection", reflection_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decision)
    graph.add_edge("sql_branch", "synthesis")
    graph.add_edge("vector_branch", "synthesis")
    graph.add_edge("synthesis", "reflection")
    graph.add_conditional_edges("reflection", reflection_decision)

    return graph.compile()


def run_agent(query: str) -> str:
    result = run_agent_full(query)
    return result["draft_answer"]


def run_agent_full(query: str) -> dict:
    """Same as run_agent, but returns the whole final state — needed for
    eval, since RAGAS scoring needs the context the answer was based on,
    not just the answer text."""
    app = build_graph()
    initial_state: AgentState = {
        "query": query,
        "route": None,
        "sql_result": None,
        "retrieved_chunks": None,
        "draft_answer": None,
        "is_grounded": None,
        "feedback": None,
        "retry_count": 0,
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    test_query = (
        "What's my account balance for AC00001, and what's your policy "
        "on balance holds?"
    )
    print(run_agent(test_query))