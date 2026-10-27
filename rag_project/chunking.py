"""读 txt 文件 + 切分。

纯本地逻辑，不调用任何外部服务 —— 所以可以离线单测，零 API 成本。
直接运行本文件：python rag_project/chunking.py
"""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, DOC_DIR, MIN_CHUNK_LEN

# 中文必须自定义分隔符：默认分隔符不含中文标点，会把整段当一块
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def load_chunks(folder=DOC_DIR, verbose=True):
    """把一个目录下所有 txt 切成片段列表。

    返回 list[dict]，每项含 id / text / source / chunk_index。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )

    chunks = []
    for path in sorted(Path(folder).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        if verbose:
            print(f"读取 {path.name}，{len(text)} 字")

        for i, piece in enumerate(splitter.split_text(text)):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_LEN:
                continue
            chunks.append({
                "id": f"{path.stem}#{i}",
                "text": piece,
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
