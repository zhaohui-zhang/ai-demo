"""重排：向量粗排之后，用词面匹配把"字面最像"的候选提上来。

纯本地逻辑，可离线单测，零 API 成本。
"""

import jieba
from rank_bm25 import BM25Okapi

# 极简停用词：只为压掉高频虚词，别指望它替代正式停用词表
STOP = set("的了是在和与及或就都也很不有对于把被从而且这那你我他它们个什么怎么多少")


def tokenize(text):
    """中文分词 + 去单字 + 去虚词。"""
    return [w for w in jieba.lcut(text) if len(w) > 1 and w not in STOP]


def rerank_bm25(question, items, top_n=3, k=60):
    """BM25 词面重排 + RRF 融合，返回新的候选列表（不改原列表）。

    RRF: score = 1/(k+向量名次) + 1/(k+BM25名次)，k=60 是平滑常数，
    防止"某一侧排第一"就压倒性胜出 —— 单靠哪一侧都不稳，融合才稳。
    """
    if not items:
        return []
    if len(items) == 1:
        # 只有一条时无需排序，但字段契约必须一致：调用方会读 rrf/bm25
        return [dict(items[0], bm25=0.0, rrf=1 / (k + 1))]

    corpus = [tokenize(c["text"]) for c in items]
    scores = BM25Okapi(corpus).get_scores(tokenize(question))

    vec_rank = {i: r for r, i in enumerate(
        sorted(range(len(items)), key=lambda i: items[i]["distance"]), 1)}
    bm_rank = {i: r for r, i in enumerate(
        sorted(range(len(items)), key=lambda i: -scores[i]), 1)}

    out = []
    for i, c in enumerate(items):
        out.append(dict(
            c,
            bm25=float(scores[i]),
            rrf=1 / (k + vec_rank[i]) + 1 / (k + bm_rank[i]),
        ))
    return sorted(out, key=lambda x: (-x["rrf"], -x["bm25"], x["distance"]))[:top_n]

