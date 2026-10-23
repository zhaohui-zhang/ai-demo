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
CHAT_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")


def embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(
    name="demo_docs",
    metadata={"hnsw:space": "cosine"},
)

if collection.count() == 0:
    print("知识库为空，正在初始化...")
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
    print(f"初始化完成，共 {collection.count()} 条\n")


def retrieve(question, k=3):
    q_vec = embed([question])[0]
    res = collection.query(query_embeddings=[q_vec], n_results=k)

    items = []
    for i in range(len(res["ids"][0])):
        items.append({
            "text": res["documents"][0][i],
            "source": res["metadatas"][0][i]["source"],
            "distance": res["distances"][0][i],
        })
    return items


def build_prompt(question, contexts):
    ctx = ""
    for i, c in enumerate(contexts, start=1):
        ctx += f"[{i}]（来源：{c['source']}）{c['text']}\n\n"

    return f"""请仅根据下面提供的资料回答问题。

要求：
1. 只使用资料中的信息，不要编造资料里没有的内容
2. 如果资料不足以回答，就直接说"根据现有资料无法回答"
3. 回答末尾用「依据：[编号]」标出你使用了哪几条资料

资料：
{ctx}
问题：{question}"""


def rag(question, k=3):
    print("=" * 55)
    print(f"❓ 问题：{question}\n")

    contexts = retrieve(question, k)

    print("📚 检索到的资料：")
    for i, c in enumerate(contexts, start=1):
        print(f"  [{i}] 距离={c['distance']:.4f} | {c['source']} | {c['text'][:40]}...")

    prompt = build_prompt(question, contexts)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    print("\n💬 回答：")
    print(resp.choices[0].message.content)
    print()


rag("我最近总是睡不着，有什么办法？")
rag("红烧肉怎么做？")
rag("量子计算机的原理是什么？")
