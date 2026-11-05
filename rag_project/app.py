"""Web 界面：知识库问答 Demo。

运行（在 ai-demo 目录下）：
    myenv\\Scripts\\streamlit.exe run rag_project/app.py

界面分三块：
    左栏  知识库状态 + 检索参数（阈值可拖动，现场演示门控效果）
    中间  对话区
    回答下 检索细节（每条候选的距离、是否通过门控）
"""

import sys
from pathlib import Path

# Streamlit 把「你敲命令的目录」当工作目录，不一定是 rag_project。
# 不把脚本所在目录塞进 sys.path，from config import ... 就会 ModuleNotFoundError。
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chromadb  # noqa: E402
import streamlit as st  # noqa: E402

from config import (  # noqa: E402
    CHAT_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    MAX_DISTANCE,
    REFUSE_ANSWER,
    TOP_K,
)
from indexing import build_index  # noqa: E402
from qa import ask  # noqa: E402

st.set_page_config(
    page_title="知识库问答 · RAG",
    page_icon="📚",
    layout="wide",
)

EXAMPLES = (
    "巴沙鱼要蒸多久？",
    "我晚上总是睡不着",
    "chunk_overlap 是什么？",
    "红烧肉怎么做？",
    "量子计算机的原理是什么？",
)


@st.cache_resource(show_spinner=False)
def load_collection():
    """整个会话只连一次库。

    st.cache_resource 缓存的是「对象」而不是「数据」，适合放数据库连接、
    模型这类重量级资源。每次交互都重新 new 一个 client 会很慢。
    """
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return chroma.get_collection(COLLECTION_NAME)
    except Exception:
        return None  # 还没建过索引


def render_answer(res):
    """回答主体。三种 reason 对应三种不同的提示，不能混成一句。"""
    if res["reason"] == "no_hit":
        st.warning("没有检索到任何片段：限定数据源后为空，或知识库里确实没有相关内容。")
    elif res["reason"] == "over_threshold":
        st.warning(
            f"检索到的 {len(res['candidates'])} 条片段距离全部超过阈值 "
            f"{res['threshold']}，判定为「资料不足」——**未调用模型，直接拒答**。"
        )
    else:
        st.markdown(res["answer"])


def render_stats(res):
    cols = st.columns(4)
    cols[0].metric("耗时", f"{res['elapsed']:.2f}s")
    cols[1].metric("候选片段", len(res["candidates"]))
    cols[2].metric("通过门控", len(res["used"]))
    cols[3].metric("调用模型", "是" if res["model_called"] else "否")


def render_candidates(res):
    """检索细节默认收起：主界面要干净，但细节必须能查。"""
    if not res["candidates"]:
        return

    passed = len(res["used"])
    total = len(res["candidates"])
    with st.expander(f"检索细节：{total} 条候选，{passed} 条通过门控"):
        for i, c in enumerate(res["candidates"], start=1):
            ok = c["distance"] <= res["threshold"]
            badge = "✅ 通过" if ok else "⛔ 超阈值"
            st.markdown(
                f"**[{i}] {c['source']}** · 距离 `{c['distance']:.4f}` · {badge}"
            )
            # 进度条画「相似度」（1 - 距离），比画距离更符合直觉：条越长越相关
            st.progress(
                max(0.0, min(1.0, 1 - c["distance"])),
                text=f"相似度 {1 - c['distance']:.2f}",
            )
            st.caption(c["text"])
            if i < total:
                st.divider()


collection = load_collection()

# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.title("📚 知识库问答")
    st.caption(f"生成模型 `{CHAT_MODEL}` · 向量模型 `{EMBED_MODEL}`（1024 维）")

    if collection is None:
        st.error("索引不存在，请点下面的按钮建库")
        st.metric("库内片段", 0)
        sources = []
    else:
        st.metric("库内片段", collection.count())
        metadatas = collection.get()["metadatas"] or []
        sources = sorted({m["source"] for m in metadatas})
        st.caption("数据源：" + "、".join(sources))

    if st.button("重建索引", type="primary", use_container_width=True):
        with st.spinner("重建中，请稍候…"):
            build_index(verbose=False)
        load_collection.clear()  # 缓存的是旧 collection 对象，必须清掉
        st.rerun()

    st.divider()
    st.subheader("检索参数")
    top_k = st.slider("检索条数 top-k", 1, 8, TOP_K)
    threshold = st.slider(
        "距离阈值",
        min_value=0.20,
        max_value=0.90,
        value=MAX_DISTANCE,
        step=0.01,
        help="越小越严格：更多问题会被拒答；越大越宽松：答得多但可能瞎编。",
    )
    source = st.selectbox("限定数据源", ["全部"] + sources)
    where = None if source == "全部" else {"source": source}
    use_rerank = st.checkbox(
        "BM25 重排",
        value=False,
        help="先粗排 9 条，门控过滤后用 BM25+RRF 精排，最后仍取 top-k 进 prompt",
    )

    st.divider()
    st.subheader("试这些问题")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True, key=f"ex::{q}"):
            st.session_state.pending = q

    st.divider()
    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- 主区 ----------------
st.title("知识库问答 Demo")
st.caption(
    "Chroma 向量检索 + 距离门控 + 大模型生成。"
    "回答只依据左侧数据源中的文档，资料不足时拒答。"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_answer(msg["result"])
            render_stats(msg["result"])
            render_candidates(msg["result"])

typed = st.chat_input("问点什么，例如：巴沙鱼要蒸多久？")
prompt = st.session_state.pop("pending", None) or typed

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if collection is None:
            st.error("索引还没建好，请先在左侧点「重建索引」。")
        else:
            with st.spinner("检索中…"):
                res = ask(collection, prompt, k=top_k, where=where,
                          max_distance=threshold, use_rerank=use_rerank)
            res["threshold"] = threshold  # 记下这次用的阈值，历史消息回看时才对得上

            render_answer(res)
            render_stats(res)
            render_candidates(res)
            st.session_state.messages.append(
                {"role": "assistant", "content": res["answer"], "result": res}
            )

with st.expander("这一页背后发生了什么"):
    st.markdown(
        f"""
1. 你输入的句子 → `embed()` → 1024 维向量
2. `collection.query(...)` → 返回 top-{TOP_K} 候选（含 `distances`）
3. **门控**：距离 > 阈值的片段丢掉。剩下为空 → 直接拒答，**不调模型**
4. `build_prompt()`：把片段编号 + 标来源，拼进 prompt
5. `chat.completions.create(temperature=0)` → 回答
6. 回答末尾的「依据：[n]」对应第 4 步的编号

拒答话术：`{REFUSE_ANSWER}`
"""
    )
