"""集中配置：所有常量 + 外部客户端。

全项目只有这一个文件读 .env、创建 client。
换模型、换 Key、调阈值，只改这里。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------- 路径 ----------
# 用脚本自身的绝对位置定位项目根，这样从任何目录运行都能找到数据
ROOT_DIR = Path(__file__).resolve().parent.parent
DOC_DIR = ROOT_DIR / "knowledge"
CHROMA_DIR = ROOT_DIR / "chroma_db"

# ---------- 文档格式 ----------
# 支持的扩展名。以后加 .pdf，只改这一行 + chunking.iter_doc_paths
DOC_EXTENSIONS = (".txt", ".md")

# ---------- 模型 ----------
EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024  # text-embedding-v3 输出维度，建 collection 时必须与之一致
CHAT_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")

# ---------- 检索 ----------
COLLECTION_NAME = "kb_docs"
BATCH_SIZE = 10      # text-embedding-v3 单次 input 上限
TOP_K = 3            # 每次检索取几条
MAX_DISTANCE = 0.45  # 距离门控阈值：实测真相关 0.29~0.37，假相关 0.51~0.71

# ---------- 切分 ----------
CHUNK_SIZE = 170     # 单块最大字符数：约等于一道完整菜谱的长度
CHUNK_OVERLAP = 30   # 相邻块共享字符数，防答案被切断在边界
MIN_CHUNK_LEN = 10   # 低于此长度的碎片直接丢弃

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url=os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
)
