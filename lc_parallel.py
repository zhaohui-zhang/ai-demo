import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

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


def clean(s):
    return str(s).strip()


text = """这款无线耳机我用了两周。音质在同价位里算不错，低音有点闷。
降噪效果一般，地铁里还是能听到报站声。续航倒是很给力，一次充电能用两天。
佩戴舒适度不错，戴久了耳朵不疼。缺点是充电盒有点大，放裤兜里硌得慌。
总体来说是性价比之选，适合预算有限又想要降噪的人。"""

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是编辑。用不超过 50 字概括核心内容，不要任何开场白。"),
    ("user", "{text}"),
])

keywords_prompt = ChatPromptTemplate.from_messages([
    ("system", "从文本中提取 3 个关键词，用顿号分隔，只输出关键词本身。"),
    ("user", "{text}"),
])

parallel = RunnableParallel(
    summary=summary_prompt | llm | StrOutputParser() | RunnableLambda(clean),
    keywords=keywords_prompt | llm | StrOutputParser() | RunnableLambda(clean),
)

result = parallel.invoke({"text": text})

print("===== 并行结果 =====")
print("返回类型：", type(result).__name__)
print("\n摘要：", result["summary"])
print("关键词：", result["keywords"])
texts = [
    "这本书讲的是明朝历史，作者是当年明月，语言很幽默。",
    "我昨天去爬山了，山顶风景特别好，就是人有点多。",
    "Python 的列表推导式写起来很简洁，但可读性要注意。",
]

inputs = [{"text": t} for t in texts]
results = parallel.batch(inputs)

print("\n===== 批量结果 =====")
for i, r in enumerate(results, start=1):
    print(f"\n[{i}] 摘要：{r['summary']}")
    print(f"    关键词：{r['keywords']}")
