import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    ("system", "你是一位{role}，回答简洁，不超过三句话。"),
    ("user", "{question}"),
])

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "role": "资深的 Python 工程师",
    "question": "用一句话解释什么是装饰器。",
})

print("===== 结果 =====")
print(result)
print("\n类型：", type(result).__name__)

print("\n--- 兼容性测试 ---")
print("能否取长度：", len(result))
print("能否切片：", result[:10])
