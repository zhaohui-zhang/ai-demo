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

target = "汤有点咸，但老板特别热情。"

zero_shot = f"""判断下面这句评论的情感，用一个词概括。

评论：{target}"""

few_shot = f"""判断评论的情感，只能输出「好评」「差评」「中评」三者之一。

评论：这家餐厅的菜太好吃了！
答案：好评

评论：服务太慢，等了一个小时。
答案：差评

评论：还行吧，没什么特别的。
答案：中评

评论：{target}
答案："""

for name, prompt in [
    ("zero-shot（不给例子）", zero_shot),
    ("few-shot（给 3 个例子）", few_shot),
]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    print(f"\n===== {name} =====")
    print(resp.choices[0].message.content)
    time.sleep(4)
