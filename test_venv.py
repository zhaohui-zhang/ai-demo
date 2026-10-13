import requests
print("requests包导入成功！")
res = requests.get("https://www.baidu.com")
print(f"网页状态码：{res.status_code}")