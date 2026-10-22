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
    temperature=0.5,
)

SYSTEM = """你是一个友好的 AI 学习助手，正在帮助一位大三学生学习 AI 应用开发。
回答要求：
1. 简洁，不超过 5 句话
2. 尽量用生活中的比喻
3. 如果对方问代码问题，给出可运行的简短示例"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
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


def ask(session_id, text):
    return chat.invoke(
        {"input": text},
        config={"configurable": {"session_id": session_id}},
    )


def show_history(session_id):
    hist = store.get(session_id)
    if not hist or not hist.messages:
        print("  （这个会话还没有历史）\n")
        return
    for m in hist.messages:
        who = "你" if m.type == "human" else "AI"
        preview = m.content[:60].replace("\n", " ")
        print(f"  [{who}] {preview}")
    print()


session = "default"

print("=" * 40)
print("  AI 学习助手（带记忆）")
print("=" * 40)
print("命令：/exit 退出   /history 看历史   /switch <名字> 切换会话")
print(f"当前会话：{session}\n")

while True:
    user_input = input("你：").strip()

    if not user_input:
        continue

    if user_input in ("/exit", "/quit"):
        print("再见！")
        break

    if user_input == "/history":
        show_history(session)
        continue

    if user_input.startswith("/switch"):
        parts = user_input.split()
        if len(parts) < 2:
            print("用法：/switch 会话名\n")
            continue
        session = parts[1]
        print(f"已切换到会话：{session}\n")
        continue

    try:
        reply = ask(session, user_input)
        print(f"AI：{reply}\n")
    except Exception as e:
        print(f"❌ 出错了：{type(e).__name__} - {str(e)[:100]}\n")
