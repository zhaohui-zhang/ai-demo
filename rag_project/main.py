"""CLI 入口：建库 + 交互式问答。

运行：python rag_project/main.py
"""

from indexing import build_index
from qa import answer


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


if __name__ == "__main__":
    main()
