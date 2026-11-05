"""读文档 + 切分（支持 .txt / .md / .pdf）。

纯本地逻辑，不调用任何外部服务 —— 所以可以离线单测，零 API 成本。
直接运行本文件：python rag_project/chunking.py
"""

import re
from collections import Counter
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

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

# 正文里的页码标记，如 "【P3】"。留在文本里而不是塞 metadata：
# 一个 chunk 可能跨页，metadata 只能记一个值，而【P3】是物理事实，模型能准确引用
PAGE_RE = re.compile(r"【P(\d+)】")


def strip_repeated_lines(pages, threshold_ratio=0.6):
    """删页眉页脚：每页首 2 行 + 末 2 行中，跨页重复率超过阈值的行删掉。

    不按关键词删（"XX大学""第N章"这种词正文里也可能出现），
    而是利用"页眉页脚必然在每页同一位置重复"这个结构特征。
    """
    if len(pages) < 3:
        return pages

    cnt = Counter()
    for t in pages:
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        for ln in lines[:2] + lines[-2:]:
            cnt[ln] += 1

    bad = {ln for ln, c in cnt.items() if c >= len(pages) * threshold_ratio}
    if bad:
        print(f"  清除页眉页脚 {len(bad)} 条：{sorted(bad)[:3]}")

    return [
        "\n".join(ln for ln in t.splitlines() if ln.strip() and ln.strip() not in bad)
        for t in pages
    ]


def read_pdf(path):
    """读 PDF：逐页抽文本 → 清页眉页脚 → 丢目录页 → 每页前加【P页码】标记。

    已知边界：扫描件（无文本层）抽出来是空的，这里只跳过并提示，不做 OCR。
    """
    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "").strip() for p in reader.pages]
    pages = strip_repeated_lines(pages)

    out = []
    for i, t in enumerate(pages, 1):
        if not t:
            continue
        # 目录页：开头就是"目录"，且"第N章"这类行很多。
        # 目录关键词极密集但没有答案正文，是最典型的"检索对了却答不出"的噪声源
        if "目录" in t[:30] and t.count("第") >= 5:
            print(f"  跳过目录页 P{i}")
            continue
        out.append(f"【P{i}】{t}")

    text = "\n\n".join(out)
    zh = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ratio = zh / len(text) * 100 if text else 0
    if ratio < 10:
        print(f"  ⚠ {path.name} 中文占比仅 {ratio:.0f}%，疑似扫描件/字体缺映射，需 OCR")
    return text


def read_doc(path):
    """读一份文档；.pdf 走 PDF 分支，.md 顺带清掉 Markdown 标记。

    必须在 read_text 之前按后缀分派：PDF 是二进制，直接当 UTF-8 读会
    UnicodeDecodeError。
    """
    if path.suffix == ".pdf":
        return read_pdf(path)

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

    返回 list[dict]，每项含 id / text / source / chunk_index / page_start。
    page_start 只对 PDF 有意义：取块内第一个页码标记，txt/md 为 0。
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

        # 页码指针：一页会被切成多块，只有第一块带【Pn】标记，
        # 后续块靠指针继承页号（否则 46 段里有 37 段 page_start=0）
        current_page = 0
        for i, piece in enumerate(splitter.split_text(text)):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_LEN:
                continue

            head = PAGE_RE.match(piece)          # 只在块首才算"本块起始页"
            marks = PAGE_RE.findall(piece)       # 块内所有标记，用于推进指针
            if head:
                page_start = int(head.group(1))
            else:
                page_start = current_page
            if marks:
                current_page = int(marks[-1])

            # 继承来的块补一个页首标记，模型才能引用页码
            if not head and page_start:
                piece = f"【P{page_start}】{piece}"

            chunks.append({
                # id 用 path.name（带后缀）：同名 .txt / .md 不会撞 id
                "id": f"{path.name}#{i}",
                "text": piece,
                # source 用 path.stem（不带后缀）：给人看、给 where 用，越短越好
                "source": path.stem,
                "chunk_index": i,
                "page_start": page_start,
            })
    return chunks


if __name__ == "__main__":
    # 单独跑这个文件，只为看切片质量，不碰向量库、不花钱
    cs = load_chunks()
    print(f"\n共 {len(cs)} 段\n")
    for c in cs:
        print(f"[{c['id']}] {len(c['text'])}字 | {c['text'][:40]}...")
