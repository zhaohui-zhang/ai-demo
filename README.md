# LLM 应用开发练习

从零开始的大模型（LLM）应用开发学习记录，使用 Python + OpenAI 兼容接口。

## 项目简介

这是我的 AI 应用开发入门项目集。从最基础的 API 调用开始，逐步实现多轮对话、
参数调优实验，为后续的 RAG、Agent 开发打基础。

## 功能列表

| 文件 | 功能 | 关键知识点 |
|---|---|---|
| `llm_demo.py` | 单次调用大模型并打印回复 | SDK 调用、`base_url`、messages 结构 |
| `params_lab.py` | 对比不同 temperature 下的输出 | temperature、token 计费 |
| `chat_demo.py` | 命令行多轮对话助手 | messages 历史管理、模型无状态特性 |

## 技术栈

- Python 3.10
- `openai` —— OpenAI 兼容 SDK
- `python-dotenv` —— 环境变量管理
- 模型：Qwen（qwen-turbo），通过阿里云百炼的 OpenAI 兼容接口调用

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/[张朝晖]/ai-demo.git
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
DASHSCOPE_MODEL=qwen-turbo
```

### 4. 运行

```bash
python llm_demo.py      # 单次 API 调用
python params_lab.py    # temperature 对比实验
python chat_demo.py     # 多轮对话助手
```

## 目录结构

```
ai-demo/
├── llm_demo.py          # 单次 API 调用
├── params_lab.py        # temperature / token 实验
├── chat_demo.py         # 多轮对话助手
├── test_venv.py         # 环境连通性测试
├── requirements.txt     # 依赖清单
├── .env.example         # 环境变量模板
└── .gitignore           # Git 忽略规则
```

## 学习笔记

- **大模型是无状态的**：多轮对话靠客户端维护 `messages` 历史，并**每轮全量发送**。
- **temperature 控制输出随机性**，但"温度高 ≠ 输出一定变化"；实测中发现过平台层面的反常现象。
- **密钥管理**：API Key 一律放 `.env`，绝不写进代码、绝不提交到仓库。

## 作者

[张朝晖] · 宁夏大学 软件工程（中软国际共建项目）· 大三
