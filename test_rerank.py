import sys
sys.path.insert(0, "rag_project")
from rerank import rerank_bm25

items = [
    {"text": "睡眠：睡前两小时不看屏幕，卧室温度调低。", "source": "睡眠笔记", "distance": 0.20},
    {"text": "荷兰豆要大火炒2分钟，木耳焯水1分钟。", "source": "菜谱", "distance": 0.31},
    {"text": "chunk_size 是单块最大字符数。", "source": "AI笔记", "distance": 0.35},
]

q = "荷兰豆炒多久？"
print("重排前：", [c["source"] for c in items])
print("重排后：", [c["source"] for c in rerank_bm25(q, items)])
for c in rerank_bm25(q, items):
    print(f"  {c['source']:6s} d={c['distance']:.2f} bm25={c['bm25']:.2f} rrf={c['rrf']:.5f}")
