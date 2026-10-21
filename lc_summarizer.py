import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm = ChatOpenAI(
    model=os.environ.get("DASHSCOPE_MODEL", "qwen-plus"),
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url=os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    temperature=0.3,
)

CHUNK_SIZE = 300


def clean(s):
    return str(s).strip()


summary_prompt = ChatPromptTemplate.from_messages([
    ("system", """请总结用户提供的文字，要求：
1. 用 3 条要点，每条单独一行，以「- 」开头
2. 每条不超过 30 字
3. 只写要点本身，不要任何开场白
4. 不要复述原文，要提炼"""),
    ("user", "{text}"),
])

summarize_chain = summary_prompt | llm | StrOutputParser() | RunnableLambda(clean)


def chunk_text(text, size=CHUNK_SIZE):
    return [text[i:i + size] for i in range(0, len(text), size)]


def summarize_long(text):
    chunks = chunk_text(text)
    print(f"文本共 {len(text)} 字，切成 {len(chunks)} 块\n")

    if len(chunks) == 1:
        print("只有一块，直接总结。\n")
        return summarize_chain.invoke({"text": text})

    inputs = [{"text": c} for c in chunks]
    partials = summarize_chain.batch(inputs, config={"max_concurrency": 2})
    print(f"已完成 {len(partials)} 块总结\n")

    merged = "\n\n".join(partials)
    print(f"合并后 {len(merged)} 字，开始二次总结...\n")

    return summarize_chain.invoke({"text": merged})


article = """睡眠对学习的作用，远比多数人想象的更重要。人在清醒时接收的信息，
最初只是暂时存放在海马体中，容量有限并且容易丢失。进入深度睡眠后，
大脑会把这些信息转移到大脑皮层进行长期存储，这个过程被称为记忆巩固。
如果长期睡眠不足，记忆巩固就会受阻，学过的东西更容易忘。

睡眠还影响理解能力。在快速眼动睡眠阶段，大脑会把白天学到的零散知识
重新组合，尝试建立新的联系。很多人睡一觉之后突然想通了某个难题，
往往就是这个机制在起作用。

此外，睡眠不足会直接削弱注意力与前额叶功能。前额叶负责计划、抑制冲动
和工作记忆，是学习活动的指挥中心。睡眠被剥夺后，它的效率会明显下降，
表现为走神、健忘和情绪波动。

因此，与其牺牲睡眠去多刷两小时题，不如保证七到八小时的睡眠。
对学习而言，睡够本身就是一种高效的学习方式。"""

print("===== 最终摘要 =====")
print(summarize_long(article))
