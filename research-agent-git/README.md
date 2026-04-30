# Research Agent (LangGraph)


核心目标不是只“能跑通”，而是具备工程化结构、质量控制、评测指标和测试覆盖。

## 目录结构

```text
research-agent/
├── app/
│   ├── graph.py
│   ├── llm.py
│   ├── memory.py
│   ├── reviewer.py
│   ├── schemas.py
│   ├── state.py
│   ├── nodes/
│   └── tools/
├── eval/
│   ├── dataset.jsonl
│   └── run_eval.py
├── tests/
│   ├── test_graph.py
│   └── test_tools.py
├── data/
├── scripts/
├── main.py
├── requirements.txt
└── .env.example
```

## 亮点能力

- 状态与协议解耦：`state.py`（运行状态）和 `schemas.py`（LLM 动作协议）分离。
- 长期记忆与运行记录：`memory.py` 支持 `long_term_memory` + `run_log` + `search_log`。
- 可解释检索：`web_search.py` 输出来源域名、分数、过滤状态，并记录搜索过滤原因。
- 抓取质量控制：`web_fetch.py` 结构化返回错误类型，支持重试，并记录长度/截断信息。
- 多 Agent 调度边界：`supervisor` 有 trace、最大轮次、结束原因；`graph` 有总步数上限。
- 报告质量控制：Writer 固定报告结构，做引用检查并结合 Reviewer 评审标签。
- 评测闭环：`eval/run_eval.py` 输出 completion_rate、avg_review_score 等指标。
- 工程测试：包含 tools 和 graph 行为测试。

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 配置环境变量

```bash
cp .env.example .env
```

3. 运行单次研究

```bash
python main.py --query "对比LangGraph和AutoGen在多Agent编排上的差异" --thread-id t1 --output reports/demo.md
```

4. 运行评测

```bash
python eval/run_eval.py --dataset eval/dataset.jsonl --output eval/results.csv
```

5. 运行测试

```bash
pytest -q
```

## 简历可写描述（示例）

- 基于 LangGraph 设计并实现多 Agent 文献调研系统（Planner/Supervisor/Researcher/Writer），支持可控路由与迭代边界。
- 设计长期记忆与运行日志体系，沉淀研究结果并统计平均评审分、通过率、延迟等指标。
- 构建可解释检索与抓取管线（域名过滤、启发式排序、结构化错误返回、抓取重试、质量判定）。
- 搭建批量评测框架与自动化测试，形成“功能开发-质量评审-数据评测”的工程闭环。

