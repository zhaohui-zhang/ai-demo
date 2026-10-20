import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = ChatOpenAI(
    model=os.environ.get("DASHSCOPE_MODEL", "qwen-plus"),
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url=os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是信息抽取器。只输出一个 JSON 对象，不要任何解释文字。"),
    ("user", """从下面的文本抽取字段，输出 JSON：
name（姓名）、gender（性别）、age（年龄，数字）、
city（城市）、job（职业）、salary（月薪，数字）

文本：
{text}"""),
])

parser = JsonOutputParser()
chain = prompt | llm | parser

data = chain.invoke({
    "text": "张伟，男，28岁，软件工程师，现在在北京工作，月薪两万五。"
})

print("===== 返回类型 =====")
print(type(data).__name__)

print("\n===== 内容 =====")
print(data)

print("\n===== 直接取字段 =====")
print("姓名：", data["name"])
print("年龄：", data["age"], "→ 类型：", type(data["age"]).__name__)
print("月薪：", data["salary"], "→ 类型：", type(data["salary"]).__name__)
