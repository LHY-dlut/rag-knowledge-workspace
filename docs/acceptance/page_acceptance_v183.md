# 浏览器页面验收 V1.83（真实百炼模型）

时间：2026-10-07　执行器：`scripts/ui_e2e.cjs`（Playwright，headless）
目标：基准栈前端 `http://127.0.0.1:18790`，镜像 `zhixu-rag-api:1.82`（与冻结的 v1.83 源码
54 个 .py 文件逐字节一致），`MODEL_PROVIDER=dashscope`（qwen-plus + text-embedding-v4 + gte-rerank-v2）。

## 结果

`passed: true`，**18 项检查全部通过，0 条页面错误**，10 张截图。
`mocked_checks` 1 项：`mocked_missing_judge_values_render`（前端缺失指标渲染，用显式 mock 载荷；
后端 Judge 失败另有 pytest 覆盖）。

通过的检查：

    registration / knowledge_base / sample_upload / ready_status
    chat_final_answer_and_citations / retrieval_debug / seven_switches
    evaluation_chart / three_strategy_comparison / document_generated_dataset
    application_creation_and_chat / feedback_and_manual_dataset / model_connectivity_test
    replacement_file_and_new_revision / refresh_login_and_history
    explicit_cancel_and_draft_cleanup / refresh_active_run_recovery
    mocked_missing_judge_values_render

截图与结果 JSON 保留在本地 `production_acceptance_20261003/ui_v183_on/`，不随源码发布。

## 关键前提：本次是在开关打开的情况下通过的；该值随后成为默认

本次执行时 `scope_no_upgrade_relaxation=true` 是**显式打开**的（当时默认仍是 false）。
同一脚本在当时的默认配置下无法通过，稳定卡在"载入示例 → 提问"：
示例文档写着"员工出差后必须在7天内提交报销申请"，界面自带的示例问题就是问这个，
但系统连续 3 次生成后拒答。

轨迹显示 grade `passed: true`、check 语义判据 `supported`，
卡在范围核验`答案=permission，原文=asserted`。打开开关后同一问题正常作答并带引用
（`出差报销申请需在出差后7天内提交[S1]`）。

**2026-10-08 该规则经用户确认改为默认开启（V84）**，因此本记录同时也是默认配置下的
验收依据。详见 [refusal_attribution_v83.md](../refusal_attribution_v83.md)。

## 脚本改动

`scripts/ui_e2e.cjs` 原有超时为硬编码 15/20 秒，只适用于演示模型
（真实模型单题实测 60–190 秒）。改为 `UI_TIMEOUT_MS` 可覆盖，**默认值保持原样**，
不改变既有验收口径。

## 消耗与边界

- 2026-10-07 账本：1,144 → 1,340 次派发（含两次失败尝试、单题验证与本次成功运行）
- 单次运行，未重复采样；"模型连通性测试"只验证连通，不代表并发容量
- 未做真实模型并发压力；该门槛仍记为未执行
