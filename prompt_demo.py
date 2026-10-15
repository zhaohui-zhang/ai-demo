import os
import time
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

question = "介绍一下 Python 是什么。"

personas = [
    None,
    "你是一个 5 岁的小朋友，说话简短可爱，用词非常简单。",
    "你是一位严肃的大学教授，要用学术语言，表述严谨。",
    "你是一个只会说文言文的古人，回答必须用文言文。",
]

for p in personas:
    messages = []
    if p:
        messages.append({"role": "system", "content": p})
    messages.append({"role": "user", "content": question})

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    )
    print(f"\n===== system：{p or '（未设置）'} =====")
    print(resp.choices[0].message.content)

    time.sleep(4)
 