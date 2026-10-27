"""评估：跑测试集，算命中率 / 拒答准确率 / 边界通过率。

运行：python rag_project/evaluate.py

判定规则：
    该答的题（answer/boundary）：回答里不能出现拒答话术，且必须说到 expect_any 里任一关键词
    该拒的题（refuse）：回答里必须出现拒答话术
"""

import json
import time
from pathlib import Path

from config import EVAL_FILE
from indexing import build_index
from qa import answer

REFUSE = "根据现有资料无法回答"


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


def rate(rows, group):
    """算某一组的通过率，返回 (通过数, 总数)。"""
    sub = [r for r in rows if r["case"]["group"] == group]
    return sum(1 for r in sub if r["ok"]), len(sub)


def report(rows):
    """打印汇总指标和失败清单。"""
    print("\n" + "=" * 55)
    print("评估结果")
    print("=" * 55)

    labels = [
        ("answer", "A 该答命中率"),
        ("boundary", "B 边界通过率"),
        ("refuse", "R 拒答准确率"),
    ]
    for group, label in labels:
        ok, total = rate(rows, group)
        pct = ok / total * 100 if total else 0
        print(f"{label:14s} {ok}/{total} = {pct:.1f}%")

    ok_all = sum(1 for r in rows if r["ok"])
    print(f"{'总通过率':14s} {ok_all}/{len(rows)} = {ok_all / len(rows) * 100:.1f}%")

    avg = sum(r["cost"] for r in rows) / len(rows)
    print(f"{'平均耗时':14s} {avg:.2f}s")

    # 分组耗时能直接看出门控省了多少：被拦下的题不调模型，耗时应该接近 0
    print("\n分组平均耗时：")
    for group, label in labels:
        sub = [r["cost"] for r in rows if r["case"]["group"] == group]
        if sub:
            print(f"  {label:14s} {sum(sub) / len(sub):.2f}s")

    fails = [r for r in rows if not r["ok"]]
    if fails:
        print("\n失败清单：")
        for r in fails:
            print(f"  {r['case']['id']} {r['case']['question']}")
            print(f"    原因：{r['why']}")
            print(f"    回答：{r['reply'][:60].replace(chr(10), ' ')}")
    else:
        print("\n全部通过，没有失败项。")


def main():
    cases = load_cases()
    print(f"测试集共 {len(cases)} 道题")

    collection = build_index(verbose=False)
    if not collection:
        return

    rows = []
    for case in cases:
        t0 = time.time()
        reply = answer(collection, case["question"], verbose=False)
        cost = time.time() - t0

        ok, why = judge(case, reply)
        rows.append({"case": case, "ok": ok, "why": why, "cost": cost, "reply": reply})

        mark = "通过" if ok else "未过"
        print(f"{mark} {case['id']} {case['question']} ({cost:.1f}s, {why})")

    report(rows)


if __name__ == "__main__":
    main()
