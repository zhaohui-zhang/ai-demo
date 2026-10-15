import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

for m in ["qwen-turbo", "qwen-plus"]:
    try:
        r = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "只回复两个字：可以"}],
            max_tokens=10,
        )
        print(f"✅ {m} 正常：{r.choices[0].message.content}")
    except Exception as e:
        print(f"❌ {m} 失败：{type(e).__name__} → {str(e)[:100]}")
