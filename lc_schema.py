import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
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


class Person(BaseModel):
    name: str = Field(description="姓名")
    gender: str = Field(description="性别")
    age: int = Field(description="年龄")
    city: str = Field(description="城市")
    job: str = Field(description="职业")
    salary: int = Field(description="月薪，单位元")


parser = JsonOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是信息抽取器。{format_instructions}"),
    ("user", "{text}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser

data = chain.invoke({
    "text": "张伟，男，28岁，软件工程师，现在在北京工作，月薪两万五。"
})

print(data)
print("年龄类型：", type(data["age"]).__name__)
