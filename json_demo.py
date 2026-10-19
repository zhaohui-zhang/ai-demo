import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url=os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
)
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")

text = """张伟，男，28岁，软件工程师，现在在北京工作，月薪两万五。
他毕业于宁夏大学，专业是软件工程，2019年毕业。"""

prompt = f"""从下面的文本中抽取信息，只输出一个 JSON 对象。
不要输出任何解释文字，也不要加 ```json 代码块标记。

JSON 格式如下：
{{
  "name": "姓名",
  "gender": "性别",
  "age": 年龄数字,
  "city": "城市",
  "job": "职业",
  "salary": 月薪数字,
  "school": "毕业院校",
  "major": "专业",
  "graduation_year": 毕业年份数字
}}

文本：
{text}"""

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
)

raw = resp.choices[0].message.content

print("===== ① 模型原始输出 =====")
print(raw)

print("\n===== ② 尝试解析 =====")
try:
    data = json.loads(raw)
    print("✅ 解析成功，得到的是 Python 字典：")
    print(data)
    print(f"\n姓名：{data['name']}")
    print(f"年龄：{data['age']}（类型：{type(data['age']).__name__}）")
    print(f"城市：{data['city']}")
except json.JSONDecodeError as e:
    print(f"❌ 解析失败：{e}")
