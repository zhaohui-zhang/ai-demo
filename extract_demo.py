import os
import json
import time
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

texts = [
    "李明，男，22岁，应届毕业生，专业是计算机科学，毕业于西安交通大学。",
    "王芳，女，35岁，产品经理，在上海工作，毕业于复旦大学。",
    "这是一本讲历史的书，作者不太出名。",
]


def extract(text):
    prompt = f"""从文本中抽取信息，只输出一个 JSON 对象，不要输出任何其他内容。

JSON 格式：
{{
  "name": "姓名，没有就填 null",
  "gender": "性别，男或女，没有就填 null",
  "age": 年龄数字，没有就填 null,
  "job": "职业，没有就填 null",
  "city": "城市，没有就填 null",
  "school": "毕业院校，没有就填 null"
}}

文本：
{text}"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content


def clean_json(s):
    s = s.strip()
    if s.startswith("```"):
        parts = s.split("\n")[1:]
        if parts and parts[-1].strip().startswith("```"):
            parts = parts[:-1]
        s = "\n".join(parts)
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1:
        s = s[start:end + 1]
    return s


results = []

for i, t in enumerate(texts, start=1):
    print(f"\n--- 第 {i}/{len(texts)} 条 ---")
    print("原文：", t)
    try:
        raw = extract(t)
        data = json.loads(clean_json(raw))
        print("✅ 成功：", data)
        results.append(data)
    except json.JSONDecodeError as e:
        print("❌ JSON 解析失败：", e)
        results.append({"error": "json_parse_failed", "raw": raw})
    except Exception as e:
        print("❌ 调用失败：", type(e).__name__, e)
        results.append({"error": "call_failed", "detail": str(e)})
    time.sleep(4)

with open("extracted.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n处理完毕：共 {len(results)} 条，已写入 extracted.json")
