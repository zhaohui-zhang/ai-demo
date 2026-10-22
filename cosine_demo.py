import os
import math
from dotenv import load_dotenv
from openai import OpenAI


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


print("===== Part A：小向量手工验证 =====")

cases = [
    ("[1,2] vs [2,1]", [1, 2], [2, 1]),
    ("[1,2] vs [-2,-1]", [1, 2], [-2, -1]),
    ("[1,0] vs [0,1]", [1, 0], [0, 1]),
    ("[1,2] vs [1,2]", [1, 2], [1, 2]),
]

for name, x, y in cases:
    print(f"{name:22s} → {cosine_sim(x, y):.4f}")


load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url=os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
)

texts = [
    "今天天气很好，适合出去散步。",
    "外面阳光明媚，出门走走吧。",
    "这道菜放了太多辣椒，太辣了。",
]

resp = client.embeddings.create(model="text-embedding-v3", input=texts)
vecs = [d.embedding for d in resp.data]

print("\n===== Part B：真实文本的相似度 =====")
for i in range(len(vecs)):
    for j in range(i + 1, len(vecs)):
        s = cosine_sim(vecs[i], vecs[j])
        print(f"句子{i+1} ↔ 句子{j+1}  →  {s:.4f}")

print("\n原文对照：")
for i, t in enumerate(texts, start=1):
    print(f"  句子{i}：{t}")
