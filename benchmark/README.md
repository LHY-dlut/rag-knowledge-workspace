# 公开基准评测工具（CRUD-RAG / RGB）

本目录包含公开基准（CRUD-RAG、RGB）评测的两件工具。评测协议见本机
`evaluation/public_benchmarks_20261002/PROTOCOL.md`；冻结数据与人工确认文件
不随源码分发（CRUD 官方语料未找到许可文件，不认定再分发授权）。

## 1. 人工标签核对工具

```bash
python build_review_tool.py
```

生成单文件 `human_review_tool.html`（内嵌 168 个待核对条目与原文）。
人工核对：确认/修改官方答案 → 在原文中框选证据区间 → 逐条确认 →
导出 `human_review_confirmed_v1.json`，保存到冻结数据目录的 `prepared_v2/`。

协议要求：证据区间必须由人工从原文选择，脚本不得代签。

## 2. CRUD 三档策略对比执行器

```bash
# 冻结数据目录通过环境变量指定（默认 ../prepared_v2）
export BENCH_PREPARED_DIR=/path/to/public_benchmarks_20261002/prepared_v2

python run_crud_benchmark.py \
  --api-base http://127.0.0.1:18790 \
  --username bench_admin --password <password> \
  --review-file /path/to/human_review_confirmed_v1.json \
  --credentials-csv /path/to/credentials.csv \
  --pg-container <compose项目名>-postgres-1 \
  --check-only          # 先校验人工确认文件完整性，不调用任何服务
```

流程：校验人工确认 → 注册/登录 → 创建隔离评测知识库 → 上传 314 份
CRUD 语料 → 等待入库并冻结索引快照 → 真实模型把官方答案分解为原子事实
（温度 0，逐条验证可在证据原文逐字定位）→ 通过 PostgreSQL 容器把人工证据
区间绑定到真实子块 ID → 按 `prepared_v2/protocol.json` 的
`fair_retrieval_configs` 逐档运行 dense/hybrid/full（分批防 300s 超时）→
导出报告 JSON（四指标、拒答率、辅助 P/R/F1、缺失清单、耗时、索引快照）。

公平检索三档使用相同预算（candidate_k=16、context_k=4、context_chars=10000），
仅切换向量、BM25/RRF、重排组件；改写/多查询/HyDE 全部关闭。
