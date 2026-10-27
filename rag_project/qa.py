"""检索 + 门控 + 生成。

这是 RAG 的核心三段：retrieve → build_prompt → answer。
"""

from config import CHAT_MODEL, MAX_DISTANCE, TOP_K, client
from indexing import embed

PROMPT_TEMPLATE = """请根据下面提供的资料回答问题。

要求：
1. 优先使用资料中的信息回答；资料没有直接答案时，可以基于资料里的相关内容做合理归纳，但不要补充资料之外的具体事实（如具体药物、剂量、品牌）
2. 只有资料与问题完全无关时，才回答"根据现有资料无法回答"
3. 回答末尾用「依据：[编号]」标出你使用了哪几条资料

资料：
{ctx}
问题：{question}"""


def retrieve(collection, question, k=TOP_K):
    """返回候选片段列表（未过滤）。"""
    k = min(k, collection.count())  # n_results 超过库内条数会直接抛错
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
    return PROMPT_TEMPLATE.format(ctx=ctx, question=question)


def answer(collection, question, k=TOP_K, verbose=True):
    candidates = retrieve(collection, question, k)

    if verbose:
        print("=" * 55)
        print(f"问题：{question}")
        print("检索候选：")
        for i, c in enumerate(candidates, start=1):
            flag = "通过" if c["distance"] <= MAX_DISTANCE else "超阈值"
            print(f"  [{i}] 距离={c['distance']:.4f} {flag} | "
                  f"{c['source']} | {c['text'][:24]}...")

    # 门控：距离超阈值的片段一律不进 prompt
    contexts = [c for c in candidates if c["distance"] <= MAX_DISTANCE]
    if not contexts:
        if verbose:
            print(f"\n回答：\n根据现有资料无法回答"
                  f"（检索距离全部超过阈值 {MAX_DISTANCE}，未调用模型）\n")
        return "根据现有资料无法回答"

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": build_prompt(question, contexts)}],
        temperature=0,
    )
    text = resp.choices[0].message.content

    if verbose:
        print("\n回答：")
        print(text)
        print()
    return text
