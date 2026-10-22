import os
import chromadb
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


def embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


chroma = chromadb.PersistentClient(path="./chroma_db")

collection = chroma.get_or_create_collection(
    name="demo_docs",
    metadata={"hnsw:space": "cosine"},
)

docs = [
    {"id": "1", "text": "提升睡眠质量的方法：保持规律作息，睡前避免使用手机，卧室保持黑暗安静。", "source": "健康笔记"},
    {"id": "2", "text": "晚上总是睡不着怎么办？可以试试睡前泡脚、听轻音乐，或者做几分钟深呼吸。", "source": "问答记录"},
    {"id": "3", "text": "咖啡因的半衰期约 5 小时，下午三点后喝咖啡可能影响夜间入睡。", "source": "科普文章"},
    {"id": "4", "text": "红烧肉的做法：五花肉切块焯水，加冰糖炒糖色，再放生抽老抽慢炖一小时。", "source": "菜谱"},
    {"id": "5", "text": "失眠的常见原因包括压力过大、作息紊乱、以及睡前过度使用电子设备。", "source": "健康笔记"},
]

collection.add(
    ids=[d["id"] for d in docs],
    embeddings=embed([d["text"] for d in docs]),
    documents=[d["text"] for d in docs],
    metadatas=[{"source": d["source"]} for d in docs],
)

print(f"✅ 已存入 {collection.count()} 条\n")

question = "我最近总是睡不着，有什么办法？"
print(f"提问：{question}\n")

result = collection.query(
    query_embeddings=embed([question]),
    n_results=3,
)

print("===== 最相关的 3 条 =====")
for i in range(len(result["ids"][0])):
    print(f"\n[{i+1}] {result['documents'][0][i]}")
    print(f"    距离：{result['distances'][0][i]:.4f}")
    print(f"    来源：{result['metadatas'][0][i]['source']}")
