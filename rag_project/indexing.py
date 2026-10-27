"""向量化 + 建库。

embed() 是向量化的原子操作，建库和检索两处都要用，所以放在这里。
"""

import chromadb

from chunking import load_chunks
from config import (
    BATCH_SIZE,
    CHROMA_DIR,
    COLLECTION_NAME,
    DOC_DIR,
    EMBED_MODEL,
    client,
)


def embed(texts):
    """把一批文本转成向量列表。单次最多 BATCH_SIZE 条，超了报 InvalidParameter。"""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def build_index(folder=DOC_DIR, batch_size=BATCH_SIZE, verbose=True):
    """全量重建索引，返回 collection；读不到文件时返回 None。"""
    chunks = load_chunks(folder, verbose=verbose)
    if not chunks:
        print(f"在 {folder} 里没读到 txt，检查路径和文件后缀")
        return None

    print(f"共切出 {len(chunks)} 个片段")

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # 文档规模小时全量重建最安全：
    # 不先删，改过的文档会留下「旧文本的向量」，导致回答内容与来源对不上
    try:
        chroma.delete_collection(COLLECTION_NAME)
        print("清空旧索引，本次全量重建")
    except Exception:
        pass  # 首次建库时 collection 不存在，正常

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
