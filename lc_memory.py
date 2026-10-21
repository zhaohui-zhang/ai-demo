import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

llm = ChatOpenAI(
    model=os.environ.get("DASHSCOPE_MODEL", "qwen-plus"),
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url=os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    temperature=0.3,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个简洁的助手，回答不超过两句话。"),
    ("placeholder", "{history}"),
    ("user", "{input}"),
])

base_chain = prompt | llm | StrOutputParser()

store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


chat = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("===== 会话 A：第 1 轮 =====")
print(chat.invoke(
    {"input": "我叫张朝晖，正在学 Python。"},
    config={"configurable": {"session_id": "A"}},
))

print("\n===== 会话 A：第 2 轮 =====")
print(chat.invoke(
    {"input": "我叫什么名字？"},
    config={"configurable": {"session_id": "A"}},
))

print("\n===== 会话 B：第 1 轮（全新会话）=====")
print(chat.invoke(
    {"input": "我叫什么名字？"},
    config={"configurable": {"session_id": "B"}},
))
