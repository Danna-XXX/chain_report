# 产业链报告生成智能体 V3

基于 LLM 的产业链分析报告自动生成系统，采用 Planner-Executor-Critic 多步骤 Agent 架构，支持联网检索增强，实现产业链分析报告全流程自动化输出。

## 功能特点

- **LLM 逐节生成**：每个章节独立调用大模型，针对不同报告类型设计差异化 Prompt
- **多步骤 Agent 架构**：Planner 规划执行步骤，Executor 驱动内容生成，Critic 自动质量评审与修订
- **联网检索增强**：支持实时搜索补充行业数据
- **三类报告场景**：产业链整体分析、交易数据分析、企业深度分析
- **私有材料上传**：支持上传 Excel/CSV 数据文件补充分析

## 技术栈

- 后端：Python + Flask
- 前端：HTML + Bootstrap
- LLM：阿里云 DashScope（Qwen-Plus）
- 架构：Planner-Executor-Critic Agent

## 使用方式

1. 在项目根目录创建 `DASHSCOPE_API_KEY` 文件，填入阿里云 DashScope API Key
2. 安装依赖：`pip install -r backend/requirements.txt`
3. 运行：双击 `start_local.bat` 或执行 `python backend/app.py`
4. 浏览器打开 `http://localhost:5000`
