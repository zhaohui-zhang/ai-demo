"""评估：跑测试集，算命中率 / 拒答准确率 / 边界通过率。

运行：
    python rag_project/evaluate.py            # 基线 + 重排 对比（默认）
    python rag_project/evaluate.py --base     # 只跑基线
    python rag_project/evaluate.py --rerank   # 只跑重排

判定规则：
    该答的题（answer/boundary）：回答里不能出现拒答话术，且必须说到 expect_any 里任一关键词
    该拒的题（refuse）：回答里必须出现拒答话术
"""

import json
import sys
import time
from pathlib import Path

from config import EVAL_FILE, MAX_DISTANCE, REFUSE_ANSWER as REFUSE, TOP_K
from indexing import build_index
from qa import ask

# 分组标签：顺序即报表顺序
GROUPS = [
    ("answer", "A 该答命中率"),
    ("boundary", "B 边界通过率"),
    ("refuse", "R 拒答准确率"),
]


def load_cases(path=EVAL_FILE):
    """读测试集 JSON，返回题目列表。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def norm(text):
    """归一化：去掉所有空白，避免 "2分钟" 和 "2 分钟" 被当成不同答案。

    数字和单位之间有没有空格、全角还是半角，模型每次都可能不一样，
    不做归一化会让判定结果随机波动 —— 这类"裁判 bug"最容易被误判成"系统不行"。
    """
    return text.replace(" ", "").replace("\u3000", "").replace("\n", "")


def judge(case, reply):
    """判定单题对错，返回 (是否通过, 原因)。"""
    if case["group"] == "refuse":
        if REFUSE in reply:
            return True, "正确拒答"
        return False, "该拒答却作答了"

    if REFUSE in reply:
        return False, "该作答却拒答"
    if any(norm(k) in norm(reply) for k in case.get("expect_any", [])):
        return True, "命中关键信息"
    return False, "答了但没说到关键信息"


def run_cases(collection, use_rerank=False, max_distance=MAX_DISTANCE,
              k=TOP_K, verbose=True):
    """跑一遍全量测试集，返回每题的明细行。

    threshold / use_rerank 都做成参数，是为了同一套骨架能复用到阈值扫描：
    扫描曲线在外面套一层 for max_distance 就行。
    """
    rows = []
    for case in load_cases():
        t0 = time.time()
        res = ask(collection, case["question"], k=k,
                  max_distance=max_distance, use_rerank=use_rerank)
        cost = time.time() - t0

        ok, why = judge(case, res["answer"])
        rows.append({
            "case": case,
            "ok": ok,
            "why": why,
            "cost": cost,
            "reply": res["answer"],
            "reason": res["reason"],
            "called": res["model_called"],
            "n_cands": len(res["candidates"]),
            "n_used": len(res["used"]),
        })

        if verbose:
            mark = "通过" if ok else "未过"
            print(f"{mark} {case['id']} {case['question']} ({cost:.1f}s, {why})")
    return rows


def rate(rows, group):
    """算某一组的通过率，返回 (通过数, 总数)。"""
    sub = [r for r in rows if r["case"]["group"] == group]
    return sum(1 for r in sub if r["ok"]), len(sub)


def metrics(rows):
    """把明细压成指标 dict —— 报表和对比表共用，避免两处各算一遍。"""
    out = {}
    for group, _ in GROUPS:
        out[group] = rate(rows, group)

    out["all"] = (sum(1 for r in rows if r["ok"]), len(rows))
    out["avg_cost"] = sum(r["cost"] for r in rows) / len(rows)
    # 调模型次数是"门控省钱"的量化指标：拒答题不该花生成的钱
    out["n_called"] = sum(1 for r in rows if r["called"])
    out["avg_cands"] = sum(r["n_cands"] for r in rows) / len(rows)
    out["avg_used"] = sum(r["n_used"] for r in rows) / len(rows)
    return out


def _width(s):
    """显示宽度：中文字符占两列，直接 len() 对齐会歪。"""
    return sum(2 if ord(ch) > 127 else 1 for ch in s)


def pad(s, width, right=False):
    gap = max(width - _width(s), 0)
    return " " * gap + s if right else s + " " * gap


def report(rows, title="评估结果"):
    """打印单轮的汇总指标和失败清单。"""
    m = metrics(rows)
    print("\n" + "=" * 55)
    print(title)
    print("=" * 55)

    for group, label in GROUPS:
        ok, total = m[group]
        pct = ok / total * 100 if total else 0
        print(f"{pad(label, 14)}{ok}/{total} = {pct:.1f}%")

    ok_all, n = m["all"]
    print(f"{pad('总通过率', 14)}{ok_all}/{n} = {ok_all / n * 100:.1f}%")
    print(f"{pad('平均耗时', 14)}{m['avg_cost']:.2f}s")
    print(f"{pad('调用模型', 14)}{m['n_called']}/{n} 题")
    print(f"{pad('平均候选/进prompt', 14)}{m['avg_cands']:.1f} / {m['avg_used']:.1f} 条")

    fails = [r for r in rows if not r["ok"]]
    if fails:
        print("\n失败清单：")
        for r in fails:
            print(f"  {r['case']['id']} {r['case']['question']}")
            print(f"    原因：{r['why']}（reason={r['reason']}）")
            print(f"    回答：{r['reply'][:60].replace(chr(10), ' ')}")
    else:
        print("\n全部通过，没有失败项。")


def compare(base, rr):
    """并列打印基线 vs 重排，并列出结果发生变化的题。"""
    b, r = metrics(base), metrics(rr)

    def cell(pair):
        ok, total = pair
        return f"{ok}/{total}"

    print("\n" + "=" * 62)
    print("基线 vs 重排")
    print("=" * 62)
    print(pad("指标", 18) + pad("基线", 14, True) + pad("重排", 14, True) + pad("变化", 12, True))

    for group, label in GROUPS:
        delta = r[group][0] - b[group][0]
        sign = f"{delta:+d} 题" if delta else "持平"
        print(pad(label, 18) + pad(cell(b[group]), 14, True)
              + pad(cell(r[group]), 14, True) + pad(sign, 12, True))

    d_all = r["all"][0] - b["all"][0]
    print(pad("总通过率", 18) + pad(cell(b["all"]), 14, True)
          + pad(cell(r["all"]), 14, True)
          + pad(f"{d_all:+d} 题" if d_all else "持平", 12, True))

    print(pad("平均耗时", 18) + pad(f"{b['avg_cost']:.2f}s", 14, True)
          + pad(f"{r['avg_cost']:.2f}s", 14, True))
    print(pad("调用模型", 18) + pad(f"{b['n_called']} 题", 14, True)
          + pad(f"{r['n_called']} 题", 14, True))
    print(pad("平均候选/进prompt", 18)
          + pad(f"{b['avg_cands']:.1f}/{b['avg_used']:.1f}", 14, True)
          + pad(f"{r['avg_cands']:.1f}/{r['avg_used']:.1f}", 14, True))

    changed = [(x, y) for x, y in zip(base, rr) if x["ok"] != y["ok"]]
    print(f"\n结果发生变化的题：{len(changed)} 道")
    for x, y in changed:
        print(f"  {x['case']['id']} {x['case']['question'][:24]}")
        print(f"    基线：{'过' if x['ok'] else '未过'}（{x['why']}）"
              f" → 重排：{'过' if y['ok'] else '未过'}（{y['why']}）")

    # 都过但回答文本变了：说明重排换了上下文，属正常波动，要人工看有无实质提升
    moved = [(x, y) for x, y in zip(base, rr)
             if x["ok"] and y["ok"] and x["reply"] != y["reply"]]
    print(f"都通过但回答文本不同：{len(moved)} 道（重排换了上下文，需人工判断优劣）")


def main():
    mode = "both"
    if "--base" in sys.argv:
        mode = "base"
    elif "--rerank" in sys.argv:
        mode = "rerank"

    collection = build_index(verbose=False)
    if not collection:
        return

    base = rr = None
    if mode in ("both", "base"):
        print("\n--- 跑基线 ---")
        base = run_cases(collection, use_rerank=False)
        report(base, "基线（纯向量检索，top-k=3）")

    if mode in ("both", "rerank"):
        print("\n--- 跑重排 ---")
        rr = run_cases(collection, use_rerank=True)
        report(rr, "重排（粗排 9 条 → 门控 → BM25+RRF → top-3）")

    if base and rr:
        compare(base, rr)


if __name__ == "__main__":
    main()
