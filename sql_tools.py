"""
Wraps a text-to-SQL agent around the transactions database.
"""

from pathlib import Path

from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path("data/processed/transactions.db")

SQL_MODEL = "claude-sonnet-4-5"


def build_sql_agent():
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    llm = ChatAnthropic(model=SQL_MODEL, temperature=0)
    tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()
    agent = create_agent(model=llm, tools=tools)
    return agent


def query_transactions(question: str) -> str:
    agent = build_sql_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    test_question = "What is the account balance for account AC00001?"
    print(query_transactions(test_question))