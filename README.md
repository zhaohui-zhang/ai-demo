# LLM 应用开发练习

从零开始的大模型（LLM）应用开发学习记录，使用 Python + OpenAI 兼容接口。

## 项目简介

这是我的 AI 应用开发入门项目集。从最基础的 API 调用开始，逐步实现多轮对话、
参数调优、提示词工程、结构化输出与长文本处理，为后续的 RAG、Agent 开发打基础。

## 功能列表

### W1 · 基础与多轮对话

| 文件 | 功能 | 关键知识点 |
|---|---|---|
| `llm_demo.py` | 单次调用大模型并打印回复 | SDK 调用、`base_url`、messages 结构 |
| `params_lab.py` | 对比不同 temperature 下的输出 | temperature、token 计费 |
| `chat_demo.py` | 命令行多轮对话助手 | messages 历史管理、模型无状态特性 |
| `test_venv.py` | 环境连通性测试 | requests |

### W2 · Prompt 提示工程

| 文件 | 功能 | 关键知识点 |
|---|---|---|
| `prompt_demo.py` | 同一问题 × 4 种 system 人设 | system vs user、控制变量 |
| `fewshot_demo.py` | zero-shot vs few-shot 对比 | 少样本、锁定输出词汇表 |
| `cot_demo.py` | 直接回答 vs 分步推理 | 思维链 CoT |
| `json_demo.py` | 从文本抽取字段并输出 JSON | 结构化输出、`json.loads`、脏 JSON 清洗 |
| `extract_demo.py` | 批量抽取 + 容错 + 落盘 | 循环内 `try/except`、`json.dump` |
| `summarizer.py` | 长文本摘要（分块 + 两阶段） | Map-Reduce、切片、`join` |

## 技术栈

- Python 3.10
- `openai` —— OpenAI 兼容 SDK
- `python-dotenv` —— 环境变量管理
- 模型：Qwen（qwen-plus），通过阿里云百炼的 OpenAI 兼容接口调用

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/zhaohui-zhang/ai-demo.git
cd ai-demo
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

复制模板文件，填入你自己的密钥：

```bash
cp .env.example .env
```

`.env` 内容如下：

```
DASHSCOPE_API_KEY=sk-你的key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus
```

### 4. 运行

```bash
python llm_demo.py      # 单次 API 调用
python params_lab.py    # temperature 对比实验
python chat_demo.py     # 多轮对话助手
python prompt_demo.py   # system 人设对比
python fewshot_demo.py  # few-shot 对比
python cot_demo.py      # 思维链对比
python json_demo.py     # JSON 结构化输出
python extract_demo.py  # 批量信息抽取（生成 extracted.json）
python summarizer.py    # 文本总结器
```

## 目录结构

```
ai-demo/
├── llm_demo.py          # 单次 API 调用
├── params_lab.py        # temperature / token 实验
├── chat_demo.py         # 多轮对话助手
├── prompt_demo.py       # system 提示词对比
├── fewshot_demo.py      # few-shot 少样本对比
├── cot_demo.py          # 思维链对比
├── json_demo.py         # JSON 结构化输出
├── extract_demo.py      # 批量信息抽取
├── summarizer.py        # 文本总结器
├── test_venv.py         # 环境连通性测试
├── extracted.json       # 抽取结果示例
├── requirements.txt     # 依赖清单
├── .env.example         # 环境变量模板
└── .gitignore           # Git 忽略规则
```

## 学习笔记

### W1 · 基础与多轮对话

- **大模型是无状态的**：多轮对话靠客户端维护 `messages` 历史，并**每轮全量发送**。
- **temperature 控制输出随机性**，但"温度高 ≠ 输出一定变化"；实测中发现过平台层面的反常现象。
- **密钥管理**：API Key 一律放 `.env`，绝不写进代码、绝不提交到仓库。

### W2 · Prompt 提示工程

- **`system` 是规则，`user` 是任务。**&#8203; system 决定"怎么说"，不改变"说什么"。
- **few-shot 的最大价值是"锁定输出格式/词汇表"**&#8203;，让输出能被程序可靠解析。
- **CoT 把隐式推理变成显式文字**，用"草稿纸"提升多步任务正确率；但对简单任务反而增加成本。
- **JSON 是 Agent 的通信协议。**&#8203; 提示词里写"没有就填 null"，能有效抑制幻觉。
- **`try/except` 必须放在循环内**，才能把失败控制在单条，而不是让整批崩掉。
- **"限定条数" ≠ "限定内容"。**&#8203; 要让输出稳定，必须给每一条分配"角色"。

## 作者

张朝晖 · 宁夏大学 软件工程（中软国际共建项目）· 大三
