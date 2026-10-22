article = """AI Agent（智能体）是指能够自主规划、调用工具并完成多步任务的大模型应用。
它和普通聊天机器人最大的区别在于：聊天机器人只回答一次，而 Agent 会循环地
"思考—行动—观察"，直到达成目标。

一个典型的 Agent 由四部分组成。第一是模型本身，负责理解和决策。第二是工具，
比如搜索、计算器、数据库查询，让模型能"动手"。第三是记忆，保存对话历史和中间结果。
第四是规划能力，把大任务拆成小步骤。

以"帮我查一下明天北京的天气，如果下雨就提醒我带伞"为例，Agent 的处理过程是：
先理解意图，再决定调用天气工具，拿到结果后判断是否下雨，最后决定是否提醒。
整个过程由模型自己编排，而不是程序员写死。

当然，Agent 也有明显的风险。它可能陷入死循环，可能调用错误的工具，
也可能在错误的前提上越走越远。因此实际产品里通常要加步数上限、人工确认环节和结果校验。

理解 Agent 的最好方式，是亲手写一个最小版本。当你看到它能自己决定下一步做什么时，
就真正理解了它和普通程序的区别。"""


def split_fixed(text, size):  ##硬切##
    return [text[i:i + size] for i in range(0, len(text), size)]


def split_overlap(text, size, overlap):  ##带重叠##
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def split_by_paragraph(text):  ##按段落切##
    return [p.strip() for p in text.split("\n\n") if p.strip()]


print(f"原文总长：{len(article)} 字\n")

print("===== ① 硬切 size=100 =====")
for i, c in enumerate(split_fixed(article, 100), 1):
    print(f"[{i}] ({len(c)}字) {c[:40]}...")

print("\n===== ② 带重叠 size=100 overlap=20 =====")
chunks2 = split_overlap(article, 100, 20)
print(f"块数：{len(chunks2)}")
print(f"[1] {chunks2[0][:40]}...")
print(f"[2] {chunks2[1][:40]}...")
print(f"     ↑ 对比一下，第 2 块的开头是不是重复了第 1 块的结尾？")

print("\n===== ③ 按段落切 =====")
for i, c in enumerate(split_by_paragraph(article), 1):
    print(f"[{i}] ({len(c)}字) {c[:40]}...")

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
)

smart_chunks = splitter.split_text(article)

print("\n===== ④ LangChain 递归切分 =====")
print(f"块数：{len(smart_chunks)}\n")
for i, c in enumerate(smart_chunks, 1):
    print(f"[{i}] ({len(c)}字) {c}")
    print()
