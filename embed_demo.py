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

EMBED_MODEL = "text-embedding-v3"

texts = [
    "今天天气很好，适合出去散步。",
    "外面阳光明媚，出门走走吧。",
    "这道菜放了太多辣椒，太辣了。",
]

resp = client.embeddings.create(model=EMBED_MODEL, input=texts)

vectors = [d.embedding for d in resp.data]

print("===== 基本信息 =====")
print("句子数量：", len(vectors))
print("每个向量的维度：", len(vectors[0]))
print("本次消耗 tokens：", resp.usage.total_tokens)

print("\n===== 第 1 句向量的前 8 个数字 =====")
print([round(x, 4) for x in vectors[0][:8]])

print("\n===== 三个向量各自的前 5 个数字 =====")
for i, v in enumerate(vectors, start=1):
    print(f"句子 {i}：{texts[i-1]}")
    print(f"   → {[round(x, 4) for x in v[:5]]}")
