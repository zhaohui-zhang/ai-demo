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
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")

question = """一个水果店进了 3 箱苹果，每箱 24 个。
上午卖出 35 个，下午又进货 2 箱（每箱 24 个），然后卖出 40 个。
现在还剩多少个苹果？"""

direct = f"""请回答下面的问题，只输出最终的数字，不要解释。

{question}"""

cot = f"""请回答下面的问题。要求一步一步思考，写出每一步的计算过程，
最后单独用一行输出「答案：X」。

{question}"""

for name, prompt in [
    ("① 直接回答", direct),
    ("② 思维链 CoT", cot),
]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    print(f"\n===== {name} =====")
    print(resp.choices[0].message.content)
    print(f"[本次消耗 {resp.usage.total_tokens} tokens]")
    time.sleep(4)
