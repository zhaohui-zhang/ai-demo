import os
from dotenv import load_dotenv
from openai import OpenAI

# 从 .env 文件读取环境变量（密钥不写在代码里）
load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url=os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
)

resp = client.chat.completions.create(
    model=os.environ.get("DASHSCOPE_MODEL", "qwen-turbo"),
    messages=[
        {"role": "user", "content": "简单介绍RAG是什么"}
    ],
)

print("大模型返回结果：")
print(resp.choices[0].message.content)
