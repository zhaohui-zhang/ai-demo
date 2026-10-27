"""读文档 + 切分（支持 .txt / .md）。

纯本地逻辑，不调用任何外部服务 —— 所以可以离线单测，零 API 成本。
直接运行本文件：python rag_project/chunking.py
"""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOC_DIR,
    DOC_EXTENSIONS,
    MIN_CHUNK_LEN,
)

# 中文必须自定义分隔符：默认分隔符不含中文标点，会把整段当一块
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

# Markdown 标记清洗表。顺序重要：长标记必须排在短标记前面，
# 否则 "## 标题" 里的 "# " 会先被替换掉，留下一个孤儿 "#"
MD_TOKENS = ("### ", "## ", "# ", "**", "`", "- ", "* ")


def read_doc(path):
    """读一份文档，.md 顺带清掉 Markdown 标记。

    这是最简清洗：只处理最常见的几类标记。
    表格、图片 ![]()、链接 []() 暂不处理（后面用正则或专用切分器再做）。
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        for token in MD_TOKENS:
            text = text.replace(token, "")
    return text


def iter_doc_paths(folder):
    """列出目录下所有支持格式的文件，按文件名排序。"""
    paths = []
    for ext in DOC_EXTENSIONS:
        paths.extend(Path(folder).glob(f"*{ext}"))
    return sorted(paths)


def load_chunks(folder=DOC_DIR, verbose=True):
    """把一个目录下所有支持的文档切成片段列表。

    返回 list[dict]，每项含 id / text / source / chunk_index。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )

    chunks = []
    for path in iter_doc_paths(folder):
        text = read_doc(path)
        if verbose:
            print(f"读取 {path.name}，{len(text)} 字")

        for i, piece in enumerate(splitter.split_text(text)):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_LEN:
                continue
            chunks.append({
                # id 用 path.name（带后缀）：同名 .txt / .md 不会撞 id
                "id": f"{path.name}#{i}",
                "text": piece,
                # source 用 path.stem（不带后缀）：给人看、给 where 用，越短越好
                "source": path.stem,
                "chunk_index": i,
            })
    return chunks


if __name__ == "__main__":
    # 单独跑这个文件，只为看切片质量，不碰向量库、不花钱
    cs = load_chunks()
    print(f"\n共 {len(cs)} 段\n")
    for c in cs:
        print(f"[{c['id']}] {len(c['text'])}字 | {c['text'][:40]}...")
