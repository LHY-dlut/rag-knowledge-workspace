"""把三档评测报告 JSON 转成 Markdown 汇总（供仓库记录与简历引用）。

用法：
  python make_report.py bench_runner/out/crud_bench_report_YYYYmmdd-HHMMSS.json
"""

import json
import sys
from pathlib import Path

METRIC_LABELS = [
    ("valid_cases", "有效题数"),
    ("missing_cases", "缺失题数"),
    ("context_recall", "Context Recall"),
    ("context_precision", "Context Precision"),
    ("faithfulness", "Faithfulness"),
    ("answer_relevancy", "Answer Relevancy"),
    ("document_recall", "Document Recall"),
    ("refusal_rate", "拒答率"),
]


def fmt(value) -> str:
    if value is None:
        return "缺失"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    strategies = list(data["strategies"].keys())
    lines = [
        "# CRUD-RAG 公开基准 · 三档检索策略对比",
        "",
        f"生成时间：{data['generated_at']}　协议：{data['protocol']}",
        f"人工确认文件 SHA256：`{data['review_file_sha256']}`",
        "",
        "> 公平检索三档：相同候选与上下文预算，仅切换向量 / BM25+RRF / 重排组件；",
        "> 关闭改写、多查询与 HyDE。评分为真实模型（qwen-plus + text-embedding-v4 + gte-rerank-v2）。",
        "",
        "## 总体指标",
        "",
        "| 指标 | " + " | ".join(strategies) + " |",
        "|---|" + "---|" * len(strategies),
    ]
    for key, label in METRIC_LABELS:
        row = [fmt(data["strategies"][s]["aggregate"].get(key)) for s in strategies]
        lines.append(f"| {label} | " + " | ".join(row) + " |")

    lines += [
        "",
        "## 索引冻结快照",
        "",
        f"- 知识库 ID：`{data['freeze']['kb_id']}`",
        f"- 冻结时间：{data['freeze']['frozen_at']}　知识库版本：{data['freeze']['kb_revision']}",
        f"- 文档数：{len(data['freeze']['documents'])}",
        "",
    ]

    dropped = sum(len(f["dropped"]) for f in data["fact_report"])
    lines += [
        "## 标注与绑定",
        "",
        f"- 原子事实：{len(data['fact_report'])} 题，剔除无法逐字定位的事实 {dropped} 条（逐题记录见 JSON）",
        f"- 证据区间绑定问题：{len(data['bind_problems'])} 条",
        f"- 未纳入的核对条目：无证据 {len(data['cases_noev'])}、存疑 {len(data['cases_flagged'])}",
        "",
    ]
    if data["bind_problems"]:
        lines += ["绑定问题明细：", ""]
        lines += [f"- {p}" for p in data["bind_problems"][:20]]
        lines += [""]

    lines += [
        "## 逐题结果",
        "",
        "| 题号 | 策略 | 状态 | CR | CP | Faith | AR | 拒答 | 耗时ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in strategies:
        for i, r in enumerate(data["strategies"][s]["results"], 1):
            m = r.get("metrics", {})
            lines.append(
                f"| {i} | {s} | {r['status']} | {fmt(m.get('context_recall'))} | "
                f"{fmt(m.get('context_precision'))} | {fmt(m.get('faithfulness'))} | "
                f"{fmt(m.get('answer_relevancy'))} | {fmt(m.get('rejected'))} | {fmt(m.get('elapsed_ms'))} |"
            )
    lines += [
        "",
        "## 口径说明",
        "",
        "- 指标定义与平台质量评测一致：Context Recall 为被证据支持的参考事实占比；",
        "  Context Precision 为证据排名的平均精度（AP）；Faithfulness 为被引用证据支持的答案声明占比；",
        "  Answer Relevancy 为裁判 0-1 评分。缺失值单列，不补零。",
        "- 语料为 314 份去重文本的受限子集（官方附带新闻 + 200 条独立片段），",
        "  不等同完整 CRUD-RAG 语料库规模；本地适配协议与官方随机抽取协议存在差异，",
        "  成绩不能直接当作原论文复现成绩。",
    ]
    out = path.with_suffix(".md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
