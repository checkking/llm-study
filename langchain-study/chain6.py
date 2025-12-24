import sqlite3
import requests
import os
from langchain.chat_models import init_chat_model
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_classic import hub
from langchain.agents import create_agent


def get_engine_for_chinook_db():
    """Pull sql file, populate in-memory database, and create engine."""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


engine = get_engine_for_chinook_db()

db = SQLDatabase(engine)

llm = init_chat_model(
    model="doubao-1-5-pro-32k-250115",
    model_provider="openai",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.environ.get("ARK_OPENAI_API_KEY"),
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

toolkit.get_tools()

prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

assert len(prompt_template.messages) == 1
print(prompt_template.input_variables)

system_message = prompt_template.format(dialect="SQLite", top_k=5)

agent = create_agent(llm, toolkit.get_tools(), system_prompt=system_message)

example_query = "Which country's customers spent the most?"

events = agent.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()