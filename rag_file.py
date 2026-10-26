import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url=os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
)

DOC_DIR = "./knowledge"
CHUNK_SIZE = 170
CHUNK_OVERLAP = 30
EMBED_MODEL = "text-embedding-v3"
CHAT_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")
COLLECTION_NAME = "kb_docs"
BATCH_SIZE = 10
MAX_DISTANCE = 0.45


def embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def load_chunks(folder):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )

    chunks = []
    for path in sorted(Path(folder).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        print(f"读取 {path.name}，{len(text)} 字")

        for i, piece in enumerate(splitter.split_text(text)):
            piece = piece.strip()
            if len(piece) < 10:
                continue
            chunks.append({
                "id": f"{path.stem}#{i}",
                "text": piece,
                "source": path.stem,
                "chunk_index": i,
            })
    return chunks


def build_index(folder=DOC_DIR, batch_size=BATCH_SIZE):
    chunks = load_chunks(folder)
    if not chunks:
        print(f"在 {folder} 里没读到 txt，检查路径和文件名后缀")
        return None

    print(f"共切出 {len(chunks)} 个片段")

    chroma = chromadb.PersistentClient(path="./chroma_db")
    try:
        chroma.delete_collection(COLLECTION_NAME)
        print("清空旧索引，本次全量重建")
    except Exception:
        pass

    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embed([c["text"] for c in batch]),
            documents=[c["text"] for c in batch],
            metadatas=[
                {"source": c["source"], "chunk_index": c["chunk_index"]}
                for c in batch
            ],
        )
        print(f"  入库 {start + len(batch)}/{len(chunks)}")

    print(f"建库完成，库内 {collection.count()} 条\n")
    return collection


def retrieve(collection, question, k=3):
    k = min(k, collection.count())
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

    return f"""请根据下面提供的资料回答问题。

要求：
1. 优先使用资料中的信息回答；资料没有直接答案时，可以基于资料里的相关内容做合理归纳，但不要补充资料之外的具体事实（如具体药物、剂量、品牌）
2. 只有资料与问题完全无关时，才回答"根据现有资料无法回答"
3. 回答末尾用「依据：[编号]」标出你使用了哪几条资料

资料：
{ctx}
问题：{question}"""


def answer(collection, question, k=3):
    candidates = retrieve(collection, question, k)

    print("=" * 55)
    print(f"问题：{question}")
    print("检索候选：")
    for i, c in enumerate(candidates, start=1):
        flag = "通过" if c["distance"] <= MAX_DISTANCE else "超阈值"
        print(f"  [{i}] 距离={c['distance']:.4f} {flag} | {c['source']} | {c['text'][:24]}...")

    contexts = [c for c in candidates if c["distance"] <= MAX_DISTANCE]
    if not contexts:
        print(f"\n回答：\n根据现有资料无法回答（检索距离全部超过阈值 {MAX_DISTANCE}，未调用模型）\n")
        return

    prompt = build_prompt(question, contexts)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    print("\n回答：")
    print(resp.choices[0].message.content)
    print()


def main():
    collection = build_index()
    if not collection:
        return

    print("输入问题开始提问，直接回车退出\n")
    while True:
        q = input("你问：").strip()
        if not q:
            break
        answer(collection, q)


main()
