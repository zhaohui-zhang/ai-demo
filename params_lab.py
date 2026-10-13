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

question = "给一家开在大学旁边的奶茶店起 5 个名字，每个名字用一句话说明理由。"

for temp in [0, 1.5]:
    print(f"\n########## temperature = {temp} ##########")
    for i in range(3):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": question}],
            temperature=temp,
        )
        print(f"[第 {i+1} 次] {resp.choices[0].message.content}\n")