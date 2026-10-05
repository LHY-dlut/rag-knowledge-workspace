# -*- coding: utf-8 -*-
"""端到端冒烟测试：上传少量语料 → 等待 ready → PG 子块核查 → 证据区间绑定。

使用 run_crud_benchmark 的同一套函数，验证正式评测链路可跑通。
成本：仅少量 embedding 调用。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_crud_benchmark import (  # noqa: E402
    ALLOWLIST,
    Api,
    bind_case_spans,
    kb_documents,
    log,
    run_sql_in_container,
    wait_ready,
)

API_BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18790"
USERNAME = "bench_admin"
PASSWORD = "BenchAdmin2026!x"
PG_CONTAINER = "zhixu-rag-bench-postgres-1"
SAMPLE = 6


def main() -> None:
    api = Api(API_BASE)
    token = api.req("POST", "/api/auth/login", {"username": USERNAME, "password": PASSWORD})["data"]["access_token"]
    api = Api(API_BASE, token)
    log("登录成功")

    kb = api.req(
        "POST",
        "/api/knowledge-bases",
        {"name": f"smoke-{time.strftime('%m%d-%H%M')}", "description": "冒烟测试（非评测库）"},
    )["data"]
    kb_id = kb["id"]
    log(f"冒烟知识库 {kb_id}")

    allow = json.loads(ALLOWLIST.read_text(encoding="utf-8"))["documents"]
    # 优先选被官方证据引用的文档，保证后续绑定测试有效
    crud_all = json.loads(
        (Path(__file__).resolve().parents[1] / "prepared_v2" / "crud" / "cases.json").read_text(encoding="utf-8")
    )
    cited = {ev["document_id"] for c in crud_all for ev in c["official_evidence"]}
    cited_files = [a for a in allow if a["external_document_id"] in cited]
    others = [a for a in allow if a["external_document_id"] not in cited]
    sample = (cited_files + others)[:SAMPLE]
    log(f"上传 {len(sample)} 份样本 …")
    for item in sample:
        api.upload(f"/api/knowledge-bases/{kb_id}/documents", Path(item["path"]))
    ready, failed = wait_ready(api, kb_id, len(sample), max_wait=600)
    log(f"ready={len(ready)} failed={len(failed)}")
    if failed:
        for f in failed:
            log(f"  失败: {f['filename']} {f.get('error', '')}")
        raise SystemExit("存在入库失败")

    doc_ids = [d["id"] for d in ready]
    quoted = ",".join(f"'{d}'" for d in doc_ids)
    row = run_sql_in_container(
        PG_CONTAINER,
        f"SELECT DISTINCT owner_id || '|' || kb_id FROM chunk_vector WHERE doc_id IN ({quoted}) LIMIT 1;",
    ).strip()
    if not row:
        raise SystemExit("PG 无子块，入库链路异常")
    owner_id, pg_kb_id = row.split("|")
    counts = run_sql_in_container(
        PG_CONTAINER,
        f"SELECT count(*), count(DISTINCT doc_id), count(DISTINCT parent_id) FROM chunk_vector WHERE doc_id IN ({quoted});",
    ).strip()
    log(f"PG 子块: 总{counts}（子块数|文档数|父块数）")

    # 证据区间绑定：用官方证据做真实样本
    # 平台文档 UUID → 外部语料 ID：上传文件名为 doc-<sha>.txt
    ext_of = {d["id"]: Path(d["filename"]).stem for d in ready}
    by_doc = {}
    for c in crud_all:
        for ev in c["official_evidence"]:
            by_doc.setdefault(ev["document_id"], []).append(ev)
    hits = [d for d in doc_ids if ext_of.get(d) in by_doc]
    log(f"样本中 {len(hits)} 份文档被官方证据引用，测试绑定 …")
    ok = 0
    for did in hits[:3]:
        ext = ext_of[did]
        for ev in by_doc[ext][:1]:
            text = (Path(__file__).resolve().parents[1] / "prepared_v2" / "corpus" / f"{ext}.txt").read_text(encoding="utf-8")
            span = {"document_id": did, "start": ev["start"], "end": ev["end"], "quote": text[ev["start"]:ev["end"]]}
            ids, problems = bind_case_spans(api, PG_CONTAINER, owner_id, pg_kb_id, span, {})
            log(f"  {did[:16]}… 区间[{ev['start']},{ev['end']}) → {len(ids)} 个子块 {problems or ''}")
            if ids:
                ok += 1
    log(f"绑定测试: {ok}/{min(3, len(hits))} 成功")
    if ok == 0:
        raise SystemExit("绑定全失败，需排查")
    log("冒烟测试通过 ✓")
    log(f"提示：冒烟库 {kb_id} 可保留或删除；正式评测请用执行器另建库")


if __name__ == "__main__":
    main()
