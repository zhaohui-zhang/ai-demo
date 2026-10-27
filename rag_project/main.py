"""CLI 入口：建库 + 交互式问答。

运行：python rag_project/main.py
输入格式：
    红烧肉怎么做          → 在全部文档里检索
    #菜谱 红烧肉怎么做    → 只在「菜谱」这份文档里检索
"""

from indexing import build_index
from qa import answer


def parse_query(raw):
    """解析 '#菜谱 红烧肉怎么做' → ('菜谱', '红烧肉怎么做')。"""
    if raw.startswith("#"):
        head, _, rest = raw.partition(" ")
        return head[1:], rest.strip()
    return None, raw


def main():
    collection = build_index()
    if not collection:
        return

    print("输入问题开始提问；用 '#来源 问题' 可限定在某份文档里查；直接回车退出\n")
    while True:
        raw = input("你问：").strip()
        if not raw:
            break

        source, q = parse_query(raw)
        if not q:
            print("请在 #来源 后面写问题\n")
            continue

        where = {"source": source} if source else None
        answer(collection, q, where=where)


if __name__ == "__main__":
    main()
