"""检索 + 门控 + 生成。

这是 RAG 的核心三段：retrieve → build_prompt → ask。

两个对外入口，分工不同：
    ask()    返回结构化结果（dict），不打印 —— 给 Web 界面 / 调试用
    answer() 打印过程 + 返回字符串 —— 给命令行和评估脚本用
"""

import time

from config import CHAT_MODEL, MAX_DISTANCE, REFUSE_ANSWER, TOP_K, client
from indexing import embed

PROMPT_TEMPLATE = """请根据下面提供的资料回答问题。

要求：
1. 优先使用资料中的信息回答；资料没有直接答案时，可以基于资料里的相关内容做合理归纳，但不要补充资料之外的具体事实（如具体药物、剂量、品牌）
2. 只有资料与问题完全无关时，才回答"根据现有资料无法回答"
3. 回答末尾用「依据：[编号]」标出你使用了哪几条资料

资料：
{ctx}
问题：{question}"""


def retrieve(collection, question, k=TOP_K, where=None):
    """返回候选片段列表（未过滤）。where 可指定只在某份文档里检索。"""
    # where 会把候选池缩小，所以 k 的上界要按「过滤后」的条数算，
    # 不能用 collection.count()（那是全库条数，会让 n_results 越界报错）
    total = len(collection.get(where=where)["ids"])
    if total == 0:
        return []
    k = min(k, total)

    q_vec = embed([question])[0]
    res = collection.query(query_embeddings=[q_vec], n_results=k, where=where)

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


def ask(collection, question, k=TOP_K, where=None, max_distance=MAX_DISTANCE):
    """完整问答，返回 dict：

        answer       回答字符串（可能是拒答话术）
        candidates   检索到的全部候选（含超阈值的，界面上要展示"被拦掉的")
        used         真正进了 prompt 的片段
        reason       ok / no_hit（过滤后为空）/ over_threshold（全部超阈值）
        elapsed      总耗时（秒）
        model_called 是否真的调用了大模型 —— 门控省钱的直接证据

    max_distance 做成参数而不是直接用常量，是为了 Web 界面能拖动滑块
    现场演示"阈值变一点，拒答行为怎么变"。这不只是炫技：它就是你要
    给面试官看的那条曲线的来源。
    """
    t0 = time.time()
    candidates = retrieve(collection, question, k, where=where)

    # 情况一：过滤后一条都没有（where 写错，或那份文档没入库）
    if not candidates:
        return {
            "answer": REFUSE_ANSWER,
            "candidates": [],
            "used": [],
            "reason": "no_hit",
            "elapsed": time.time() - t0,
            "model_called": False,
        }

    # 情况二：有候选，但相关度全部不达标 —— 门控在这里拦下，不调模型
    used = [c for c in candidates if c["distance"] <= max_distance]
    if not used:
        return {
            "answer": REFUSE_ANSWER,
            "candidates": candidates,
            "used": [],
            "reason": "over_threshold",
            "elapsed": time.time() - t0,
            "model_called": False,
        }

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": build_prompt(question, used)}],
        temperature=0,
    )
    return {
        "answer": resp.choices[0].message.content,
        "candidates": candidates,
        "used": used,
        "reason": "ok",
        "elapsed": time.time() - t0,
        "model_called": True,
    }


def answer(collection, question, k=TOP_K, where=None, verbose=True):
    """命令行版：调用 ask()，把过程打印出来，只返回回答字符串。"""
    res = ask(collection, question, k, where=where)

    if verbose:
        print("=" * 55)
        print(f"问题：{question}")

        if res["reason"] == "no_hit":
            print(f"没检索到任何片段（where={where}），检查 source 名是否正确\n")
            return res["answer"]

        print("检索候选：")
        for i, c in enumerate(res["candidates"], start=1):
            flag = "通过" if c["distance"] <= MAX_DISTANCE else "超阈值"
            print(f"  [{i}] 距离={c['distance']:.4f} {flag} | "
                  f"{c['source']} | {c['text'][:24]}...")

        if res["reason"] == "over_threshold":
            print(f"\n回答：\n{res['answer']}"
                  f"（检索距离全部超过阈值 {MAX_DISTANCE}，未调用模型）\n")
            return res["answer"]

        print("\n回答：")
        print(res["answer"])
        print()

    return res["answer"]
