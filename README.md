# 产业链报告智能体 V3

基于 LLM 的产业链分析报告自动生成系统，采用 **Planner-Executor-Critic 多步骤 Agent 架构**，支持 Tavily 联网检索增强，实现产业链分析报告全流程自动化输出。

## 功能特性

- **Planner-Executor-Critic 架构**：生成前先规划每节写作要点（Planner），逐节生成时携带前文摘要避免重复（Executor），全文生成后自动审查并修订弱节（Critic）
- **双 LLM 分工**：硅基流动 GLM-5.1 负责规划/审查（结构化任务），阿里云 Qwen-Plus 负责正文生成（长文任务），互为备用
- **Tavily 联网检索**：优先使用 Tavily API 获取高质量行业信息，并对 top 结果进行正文内容抓取；自动降级为 DuckDuckGo / Bing
- **实时流式输出**：SSE 协议逐节推送，前端实时显示各阶段进度和已生成内容
- **多报告类型**：产业链整体报告、产业链交易报告（银行授信版）、公司具体报告
- **可编辑目录**：支持自定义章节结构
- **文件上传增强**：支持 Excel/CSV 结构化数据和 txt/md/json 私有材料

## 快速开始

### 1. 配置 API Key

复制 `.env.example` 为 `.env`，填入真实 API Key：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
ALIYUN_API_KEY=sk-your-aliyun-key        # 阿里云 DashScope（正文生成）
SILICONFLOW_API_KEY=sk-your-sf-key       # 硅基流动（Planner/Critic）
TAVILY_API_KEY=tvly-your-tavily-key      # Tavily 联网检索
```

| Key | 用途 | 获取地址 |
|-----|------|---------|
| 阿里云 DashScope | Qwen 正文生成（主力） | https://dashscope.aliyun.com/ |
| 硅基流动 | GLM-5.1 规划/审查 | https://siliconflow.cn/ |
| Tavily | 联网检索 | https://tavily.com/ |

> 三个 Key 均为可选，至少配置一个 LLM Key 即可启动（功能会降级）。

### 2. 启动服务

**Windows：**

```bash
start_local.bat
```

**手动启动：**

```bash
pip install -r backend/requirements.txt
python -m backend.app
```

访问地址：[http://localhost:5002](http://localhost:5002)

## 项目结构

```
报告生成智能体_v3/
├── backend/
│   ├── app.py                  # Flask 后端，包含 /api/generate/stream SSE 端点
│   └── services/
│       ├── key_loader.py       # 统一 API Key 读取（.env 格式）
│       ├── llm_client.py       # 双 LLM 路由（Qwen / GLM-5.1）
│       ├── report_engine.py    # Planner-Executor-Critic 核心引擎
│       ├── web_search.py       # Tavily + DDG + Bing 搜索
│       ├── report_catalog.py   # 报告类型和默认目录
│       └── material_parser.py  # Excel/CSV/txt 文件解析
├── frontend/
│   └── index.html              # 单页前端（流式 SSE + 实时进度）
├── .env                        # API Keys（不提交 Git）
├── .env.example                # Key 配置模板
├── .gitignore
└── requirements.txt
```

## Agent 架构说明

```
用户输入 → 联网检索（Tavily）
              ↓
          Planner（GLM-5.1）
          规划每节写作要点
              ↓
          Executor（Qwen-Plus）× N节
          每节携带：前文摘要 + 写作要点 + 检索数据
              ↓
          Critic（GLM-5.1）
          审查全文，找出最多3个弱节
              ↓
          Revise（Qwen-Plus）
          针对 Critic 意见重写弱节
              ↓
          输出完整 Markdown 报告
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `GET /` | GET | 前端页面 |
| `GET /api/config` | GET | 获取报告类型目录和 LLM 状态 |
| `GET /api/status` | GET | 查看服务状态和模型信息 |
| `POST /api/generate` | POST | 同步生成（等待全部完成后返回） |
| `POST /api/generate/stream` | POST | **流式生成**（SSE，推荐） |

### SSE 事件类型（`/api/generate/stream`）

```json
{"type": "stage",        "stage": "planning",  "msg": "正在规划报告结构…"}
{"type": "stage",        "stage": "searching", "msg": "正在联网检索…"}
{"type": "section_start","idx": 1, "total": 6, "title": "产业链定义与概述"}
{"type": "section_done", "idx": 1, "title": "产业链定义与概述", "content": "..."}
{"type": "stage",        "stage": "critic",    "msg": "正在审查报告质量…"}
{"type": "revision",     "idx": 2, "title": "产业链发展历程与现状"}
{"type": "complete",     "markdown": "...", "elapsed_sec": 45.2, "llm_used": true}
```

## 依赖

```
flask>=2.3.0
flask-cors>=4.0.0
pandas>=1.5.0
openpyxl>=3.1.0
```

无需 `python-dotenv`，`.env` 由内置解析器读取。
