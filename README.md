# 知序 RAG 知识工作台

**v1.81 版本**：在既有复现工程基础上完成融合、验收与修复，提供证据图表、引用报告与隔离静态网页预览，并补充事务回滚后的文件清理、孤立文件隔离和真实转换耗时。原 RAG 架构与检索、核验流程保留。使用说明见 [成果协议](docs/artifacts_v37.md) 和 [文件维护](docs/artifact_storage_v38.md)，推荐启动入口见 [本机启动说明](artifacts/deployment_v1_81/本机启动说明.md)，验收证据见 [docs/acceptance](docs/acceptance/) 目录。

面向个人知识库和小团队的 RAG 工程项目。基于已有复现工程进行验收与修复，提供 Vue 页面、双库持久化、异步入库、检索调试、带证据核验的问答、持久 SSE 与质量评测。原冻结版本和首次实验保留在本目录之外，本目录为独立生产候选版本。

**已完成真实双库及百炼联调、小样本质量验证与 24 小时长稳观测；公开基准的未见集质量评测尚未执行。** 不把 Mock/SQLite、缓存重放、固定缩减语料回归成绩当作生产吞吐、未见质量或完整官方基准成绩。

## 核心实现

- FastAPI + Tortoise ORM；MySQL 保存用户、文档、任务、对话、轨迹与评测，PostgreSQL 18 + pgvector 0.8.6 保存 1024 维向量。
- PDF/DOCX/XLSX/Markdown/TXT 解析，递归/短段/父子分块；保留原文位置和来源。
- Vector + jieba BM25，按排名 RRF 融合，保留百炼 Rerank 原始分数，回溯父块并约束上下文预算。
- 显式 LangGraph route/rewrite/retrieve/grade/generate/check/fallback；检索和生成独立重试上限。
- 文档版本与租约隔离：未发布索引不可检索，取消/过期/版本变化时禁止迟到正式答案。
- MySQL 持久运行队列与 SSE 序号事件，断线后后台执行，续订去重、重启回放与终态补偿。
- 支持限定否定回答；引用 ID、格式与逐字证据由后端校验，原文及位置映射保留，事实支持另行核验。
- 四项质量指标和带独立相关子块标签的辅助 P/R/F1；缺失和 Judge 冲突单列。

## 启动

需要可用 Docker Engine/Compose；Python 3.12+ 仅用于生成配置。初次从本目录执行：

```powershell
python -X utf8 scripts/setup_env.py --mode production
```

临时将生成的 `.env` 中 `ENABLE_REGISTRATION=true`，然后：

```powershell
docker compose -p zhixu-rag up --build -d --scale worker=2
```

打开 http://127.0.0.1:8080，创建账号后关闭注册并重新 up。默认 Demo 在真实双库上走通工程流程，不产生模型费用；接入已有百炼凭据后使用 qwen-plus、text-embedding-v4、gte-rerank-v2。密钥只保存在本地环境。

详细说明：[启动与运维](docs/启动与运维.md)、[生产验收计划](docs/production_acceptance_plan.md)。文档包含隔离端口、模型预算、演示顺序、排空停机与备份恢复。

## 验证与证据

依赖按 requirements.lock 和 frontend/package-lock.json 锁定；Docker 构建使用 Python 3.12，宿主机测试版本单独记录。pytest 覆盖权限、租约、索引发布、引用协议与 Agent 边界；前端 Node 测试覆盖 SSE 与恢复。真实双库测试需单独空数据库及显式隔离标志。

冻结 v3 的历史记录：仓库 160 项通过（13 真实双库、147 Mock/SQLite），另 1 项账本保护。固定回归集 151 单元协议终止 0；答案符合 106/146、误拒 31/148；仅 3 个有效不可答单元，不能推断整体无幻觉。官方标准与 AI 审阅、Judge 冲突和缺失均保留。该集合已用于错误分析和开发，不能称为未见验收。

本候选版的验证记录见 [docs/acceptance](docs/acceptance/)：生产验收报告、24 小时长稳结果（1,422 次采样、0 失败）与 v1.81 测试结果（713 本地 + 21 真实双库、0 失败）。公开基准（CRUD-RAG/RGB）仅完成数据准备，未见集质量评测尚未完成，不能标成通过。示例文档为合成资料，适合演示，公开评测语料和私有数据库不随源码发布。

## 项目展示

适合作为 RAG/后端工程简历项目展示：介绍双库发布协议、租约竞争与迟到答案防护、持久 SSE、否定证据误拒及引用协议诊断与修复，并展示可复现测试和逐题证据。个人贡献应按真实开发/验收过程说明，不能冒称真实客户上线、大规模并发或完全独立人工金标。

学习与架构资料见 docs/项目复现与学习指南.md 和 docs/来源与功能映射.md。尚未实测的可选 OCR、组织成员共享与公网运营能力不在当前已验收范围。
