import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url=os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
)
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-turbo")

# ① 对话历史：只先放一条 system，立好"人设"
messages = [
    {"role": "system", "content": "你是一个耐心的编程老师，回答简洁，喜欢用生活中的比喻。"}
]

total_tokens = 0
print("=== AI 对话小助手 ===（输入 exit 退出）\n")

while True:
    user_input = input("你：")

    if user_input.strip().lower() in ("exit", "quit", "退出"):
        print("再见！")
        break
    if not user_input.strip():
        continue

    # ② 把用户这句话追加进历史
    messages.append({"role": "user", "content": user_input})

    # ③ 把【完整历史】发给模型
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    reply = resp.choices[0].message.content
    print(f"AI：{reply}\n")

    # ④ 把模型的回复也追加进历史（关键！否则下一轮它就"失忆"）
    messages.append({"role": "assistant", "content": reply})

    # ⑤ 顺便观察历史增长与开销
    total_tokens += resp.usage.total_tokens
    print(f"[历史 {len(messages)} 条 | 本次 {resp.usage.total_tokens} tokens | 累计 {total_tokens} tokens]\n")
