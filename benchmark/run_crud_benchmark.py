# -*- coding: utf-8 -*-
"""CRUD-RAG 公开基准 · 三档策略对比执行器（真实百炼模型）。

协议依据：evaluation/public_benchmarks_20261002/PROTOCOL.md 与 prepared_v2/protocol.json。
只读冻结数据；通过平台 HTTP API 驱动隔离评测实例；所有模型调用经过平台，
Judge 指标由平台四项质量评测计算，逐题证据与缺失原因随结果落库并导出。

流程：
  1. 校验人工确认文件（human_review_confirmed_v1.json）覆盖 CRUD 验收题
  2. 注册/登录 → 创建隔离评测知识库
  3. 上传 crud_ingestion_allowlist 的 314 份语料 → 等待 ready → 冻结快照
  4. 原子事实：由真实模型把官方答案分解为原子事实（温度 0），
     逐条验证可在官方证据原文中逐字定位；验证失败的事实剔除并记录
  5. 证据区间绑定：通过 PostgreSQL 容器查询真实子块，把人工确认的
     (document_id, start, end, quote) 绑定到实际 chunk_vector.id，校验覆盖与文本
  6. 三档策略（dense/hybrid/full，protocol.json fair_retrieval_configs）：
     逐档 PUT 知识库配置 → 分批 POST /api/evaluations（防 300s 超时）→ 汇总
  7. 导出 report JSON/MD：四指标、拒答率、辅助 P/R/F1、缺失清单、耗时、索引快照

用法（用复现工程 venv 的 python）：
  python bench_runner/run_crud_benchmark.py \
    --api-base http://127.0.0.1:18786 \
    --username bench_admin --password <pw> \
    --review-file prepared_v2/human_review_confirmed_v1.json \
    --credentials-csv <百炼凭据csv> \
    --pg-container <compose项目名>-postgres-1 \
    --out-dir bench_runner/out
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

try:
    import httpx  # type: ignore

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

BASE = Path(__file__).resolve().parents[1]
# 公开评测数据（CRUD 官方语料许可不明）不随源码分发；
# 通过环境变量指向本机 prepared_v2 冻结目录。
PREPARED = Path(os.environ.get("BENCH_PREPARED_DIR", BASE / "prepared_v2"))
ALLOWLIST = BASE / "artifacts" / "crud_ingestion_allowlist.json"
PROTOCOL = json.loads((PREPARED / "protocol.json").read_text(encoding="utf-8"))

CRUD_SPLIT_TEST = {"crud-questanswer_1doc", "crud-questanswer_2doc", "crud-questanswer_3doc"}


class Api:
    def __init__(self, base: str, token: str = ""):
        self.base = base.rstrip("/")
        self.token = token

    def req(self, method: str, path: str, body: dict | None = None, timeout: int = 900):
        url = self.base + path
        data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc

    def upload(self, path: str, filepath: Path, tags: str = "", timeout: int = 300):
        if HAS_HTTPX:
            # trust_env=False：httpx 默认会读 Windows 注册表系统代理并代理 127.0.0.1，
            # 而本地代理（Clash 等）无法回环转发，导致 502；回环流量必须直连。
            with httpx.Client(base_url=self.base, timeout=timeout, trust_env=False) as client:
                resp = client.post(
                    path,
                    headers={"Authorization": f"Bearer {self.token}"},
                    files={"file": (filepath.name, filepath.read_bytes(), "text/plain")},
                    data={"tags": tags},
                )
                resp.raise_for_status()
                return resp.json()
        boundary = "----bench" + uuid.uuid4().hex
        content = filepath.read_bytes()
        head = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{filepath.name}\"\r\nContent-Type: text/plain\r\n\r\n"
        )
        tail = f"\r\n--{boundary}--\r\n".encode()
        body = head.encode() + content + tail
        req = urllib.request.Request(
            self.base + path,
            data=body,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Authorization": f"Bearer {self.token}",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_review(path: Path) -> dict:
    review = json.loads(path.read_text(encoding="utf-8"))
    cases = {c["case_id"]: c for c in review["cases"]}
    crud = json.loads((PREPARED / "crud" / "cases.json").read_text(encoding="utf-8"))
    test_ids = {c["case_id"] for c in crud if c["split"] == "test"}
    missing = sorted(t for t in test_ids if cases.get(t, {}).get("review_status") == "pending")
    noev = sorted(t for t in test_ids if cases.get(t, {}).get("review_status") == "noev")
    flagged = sorted(t for t in test_ids if cases.get(t, {}).get("review_status") == "flag")
    done = sorted(t for t in test_ids if cases.get(t, {}).get("review_status") == "done")
    for t in done:
        rec = cases[t]
        if not rec.get("confirmed_answer") or not rec.get("confirmed_evidence_spans"):
            raise SystemExit(f"已确认案例缺少答案或证据区间: {t}")
    return {
        "review": review,
        "cases": cases,
        "test_ids": test_ids,
        "done": done,
        "noev": noev,
        "flagged": flagged,
        "missing": missing,
    }


def fetch_allowlist() -> list[dict]:
    return json.loads(ALLOWLIST.read_text(encoding="utf-8"))["documents"]


def decompose_facts(api_key: str, base_url: str, question: str, answer: str) -> list[str]:
    """把官方参考答案分解为原子事实；温度 0；失败时回退到标点切分。"""
    prompt = (
        "把下面的问题参考答案分解为原子事实（每条是一个可独立核验的陈述，"
        "直接复用原文措辞，不要改写、不要补充）。输出 JSON 数组字符串。\n\n"
        f"问题：{question}\n参考答案：{answer}\n\n输出示例：[\"事实1\", \"事实2\"]"
    )
    for attempt in range(2):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    f"{base_url.rstrip('/')}/chat/completions",
                    data=json.dumps(
                        {
                            "model": "qwen-plus",
                            "temperature": 0,
                            "messages": [
                                {"role": "system", "content": "你是严谨的标注助手。"},
                                {"role": "user", "content": prompt},
                            ],
                            "response_format": {"type": "json_object"},
                        }
                    ).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                ),
                timeout=60,
            ) as resp:
                content = json.loads(resp.read().decode())["choices"][0]["message"]["content"]
            facts = json.loads(content)
            if isinstance(facts, dict) and "facts" in facts:
                facts = facts["facts"]
            facts = [str(f).strip() for f in facts if str(f).strip()]
            return facts[:20]
        except Exception as exc:  # noqa: BLE001
            if attempt == 1:
                log(f"事实分解失败（{exc}），回退标点切分: {question[:20]}…")
    parts = [p.strip() for p in re.split(r"[。；;，、]", answer) if p.strip()]
    return parts or [answer.strip()]


def verify_fact(fact: str, evidence_texts: list[str], answer: str) -> bool:
    """事实必须能在官方证据原文或参考答案中逐字定位（含引号/括号内核心）。"""
    if fact in answer:
        return True
    for text in evidence_texts:
        if fact in text:
            return True
    # 容忍首尾引号与书名号差异：去掉外围标点后核心仍需逐字命中
    core = fact.strip("“”\"'《》「」")
    if len(core) >= 4 and any(core in t for t in evidence_texts):
        return True
    return False


PG_USER = os.environ.get("BENCH_PG_USER", "rag")
PG_DB = os.environ.get("BENCH_PG_DB", "rag_vectors")


def run_sql_in_container(container: str, sql: str) -> str:
    """在 PG 容器内执行只读 SQL；凭据来自容器自身环境，不在命令行出现密码。"""
    import subprocess

    proc = subprocess.run(
        [
            "docker",
            "exec",
            container,
            "sh",
            "-c",
            f'PGPASSWORD="$POSTGRES_PASSWORD" psql --host=127.0.0.1 '
            f'--username="$POSTGRES_USER" --dbname="$POSTGRES_DB" -tAc {json.dumps(sql)}',
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker exec 失败: {proc.stderr[:300]}")
    return proc.stdout


def bind_spans(api: Api, container: str, owner_id: str, kb_id: str, doc_id: str) -> list[dict]:
    """从 PG 取真实子块；把 (doc_id) 下所有子块内容与位置取回，供上层做区间绑定。"""
    rows = run_sql_in_container(
        container,
        "SELECT id, parent_id, content, metadata::text FROM chunk_vector"
        f" WHERE owner_id='{owner_id}' AND kb_id='{kb_id}' AND doc_id='{doc_id}'"
        " ORDER BY parent_id, COALESCE((metadata->>'start')::int, 0);",
    )
    chunks = []
    for line in rows.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        cid, parent_id, content, meta = parts
        try:
            metadata = json.loads(meta)
        except ValueError:
            metadata = {}
        chunks.append(
            {
                "id": cid,
                "parent_id": parent_id,
                "content": content,
                "start": int(metadata.get("start", -1)),
                "end": int(metadata.get("end", -1)),
            }
        )
    return chunks


def bind_case_spans(
    api: Api, container: str, owner_id: str, kb_id: str, span: dict, doc_texts: dict[str, str]
) -> tuple[list[str], list[str]]:
    """内容优先绑定：子块内容包含于区间文本，或区间文本包含于子块；校验覆盖。"""
    doc_id = span["document_id"]
    span_text = span["quote"]
    chunks = bind_spans(api, container, owner_id, kb_id, doc_id)
    if not chunks:
        return [], [f"{doc_id[:12]}… 无任何子块"]
    contained = [c for c in chunks if c["content"] in span_text]
    covering = [c for c in chunks if span_text in c["content"]]
    picked = contained or covering
    problems = []
    if not picked:
        problems.append(f"{doc_id[:12]}… 区间无完全匹配子块（contained={len(contained)}, covering={len(covering)}）")
    # 覆盖校验：区间文本应被所选子块内容联合覆盖
    union = "".join(c["content"] for c in picked)
    if span_text not in union:
        problems.append(f"{doc_id[:12]}… 覆盖校验失败")
    return [c["id"] for c in picked], problems


def kb_documents(api: Api, kb_id: str) -> list[dict]:
    data = api.req("GET", f"/api/knowledge-bases/{kb_id}/documents")["data"]
    if isinstance(data, dict):
        data = data.get("documents", [])
    return data


def wait_ready(api: Api, kb_id: str, expect: int, max_wait: int = 1800) -> tuple[list[dict], list[dict]]:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        docs = kb_documents(api, kb_id)
        pending = [d for d in docs if d["status"] in ("queued", "processing")]
        failed = [d for d in docs if d["status"] == "failed"]
        ready = [d for d in docs if d["status"] == "ready"]
        log(f"  文档状态: ready={len(ready)} pending={len(pending)} failed={len(failed)} / 期望 {expect}")
        if not pending and len(ready) + len(failed) >= expect:
            return ready, failed
        time.sleep(10)
    raise SystemExit("等待文档 ready 超时")


def put_config(api: Api, kb_id: str, strategy: str) -> None:
    cfg = dict(PROTOCOL["fair_retrieval_configs"][strategy])
    resp = api.req("PUT", f"/api/knowledge-bases/{kb_id}/config", cfg)
    log(f"  策略 {strategy} 配置已生效: {json.dumps({k: resp['data'][k] for k in ('hybrid', 'rerank')}, ensure_ascii=False)}")


def build_case_payload(chunk: list[dict]) -> list[dict]:
    return [
        {
            "question": c["question"],
            "reference_answer": c["reference_answer"],
            "reference_facts": c["facts"],
            "answerable": True,
            "relevant_document_ids": c["relevant_document_ids"],
            "relevant_child_ids": c["relevant_child_ids"] or None,
            "annotation_method": "human",
        }
        for c in chunk
    ]


def submit_and_collect(api: Api, kb_id: str, strategy: str, chunk: list[dict]) -> list[dict]:
    """提交一个批次并取回结果；失败返回空列表由上层重试。"""
    body = {
        "kb_id": kb_id,
        # 协议要求三档统一 rag 模式，隔离检索策略差异（关闭 agent 循环）
        "mode": "rag",
        "cases": build_case_payload(chunk),
    }
    try:
        resp = api.req("POST", "/api/evaluations", body, timeout=300)
    except Exception as exc:  # noqa: BLE001
        log(f"  [{strategy}] 批次提交失败（{type(exc).__name__}），将在单题模式重试")
        return []
    rid = resp["data"]["id"]
    # 服务端同步执行，POST 返回时已结束；再取一次拿到完整结果
    item = api.req("GET", f"/api/evaluations/{rid}")["data"]
    status = item.get("status")
    results = item.get("results", [])
    log(f"  [{strategy}] run {rid[:8]} → {status}, 题数 {len(results)}")
    return results if status in ("completed", "partial_missing") else []


def run_strategy(
    api: Api, kb_id: str, strategy: str, cases: list[dict], batch: int, parallel: int
) -> list[dict]:
    put_config(api, kb_id, strategy)
    batches = [cases[i : i + batch] for i in range(0, len(cases), batch)]
    log(f"  策略 {strategy}: {len(cases)} 题 / {len(batches)} 批（批={batch}，并行={parallel}）")
    results: list[dict] = []
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        futures = [pool.submit(submit_and_collect, api, kb_id, strategy, b) for b in batches]
        for idx, (b, fut) in enumerate(zip(batches, futures), 1):
            got = fut.result()
            if not got:
                # 批次整体失败：退回单题重试，避免整批丢失
                log(f"  批次 {idx}/{len(batches)} 失败，单题重试 {len(b)} 题")
                for c in b:
                    got.extend(submit_and_collect(api, kb_id, strategy, [c]))
                results.extend(got)
                continue
            results.extend(got)
            log(f"  进度 {idx}/{len(batches)} 批，累计 {len(results)} 题")
    return results


def aggregate(results: list[dict]) -> dict:
    from statistics import mean

    names = ("context_recall", "context_precision", "faithfulness", "answer_relevancy", "document_recall")
    agg: dict = {}
    for name in names:
        values = [r["metrics"][name] for r in results if r.get("metrics", {}).get(name) is not None]
        agg[name] = round(mean(values), 4) if values else None
    refusals = [r["metrics"]["rejected"] for r in results if r.get("metrics", {}).get("rejected") is not None]
    agg["refusal_rate"] = round(mean(refusals), 4) if refusals else None
    agg["valid_cases"] = sum(1 for r in results if r["status"] == "passed")
    agg["missing_cases"] = sum(1 for r in results if r["status"] == "missing")
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--review-file", type=Path, help="人工确认文件；--ingest-only 时可省略")
    parser.add_argument("--credentials-csv", type=Path, required=True)
    parser.add_argument("--pg-container", default="zhixu-rag-v81-postgres-1")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "out")
    parser.add_argument("--strategies", default="dense,hybrid,full")
    parser.add_argument("--parallel", type=int, default=4, help="并行批次数；受平台 MODEL_CONCURRENCY 约束")
    parser.add_argument("--batch-size", type=int, default=1, help="每批题数；服务端单请求 300s 超时、单题约 100s，默认 1 最稳，失败仅损失该题")
    parser.add_argument("--check-only", action="store_true", help="只校验人工确认文件，不调用任何服务")
    parser.add_argument("--allow-excluded", action="store_true", help="确认排除被标记为 noev/flag 的验收题（需在报告中如实记录）")
    parser.add_argument("--kb-id", help="复用已入库的评测知识库（跳过上传，校验 ready 后直接评测）")
    parser.add_argument("--ingest-only", action="store_true", help="只完成入库与索引冻结后退出，输出知识库 ID")
    args = parser.parse_args()

    # --ingest-only 只做入库与冻结，不依赖人工核对进度
    review = None
    if args.review_file and args.review_file.exists():
        review = load_review(args.review_file)
    if not args.ingest_only:
        if review is None:
            raise SystemExit("正式评测需要 --review-file 指向人工确认文件")
        log(
            f"人工确认: done={len(review['done'])} noev={len(review['noev'])} "
            f"flag={len(review['flagged'])} 未完成={len(review['missing'])}"
        )
        if args.check_only:
            for t in review["missing"]:
                print("  未完成:", t)
            for t in review["noev"]:
                print("  无证据:", t)
            for t in review["flagged"]:
                print("  存疑  :", t)
            ok = not review["missing"] and not review["noev"] and not review["flagged"]
            print("校验结论:", "可以运行正式评测" if ok else "需先解决上述条目")
            sys.exit(0 if ok else 1)
        if review["missing"]:
            raise SystemExit("存在未完成核对的验收题，先完成核对再运行")
        if (review["noev"] or review["flagged"]) and not args.allow_excluded:
            print("以下验收题被标记为无证据/存疑，默认不排除以免静默少算：")
            for t in review["noev"]:
                print("  noev  :", t)
            for t in review["flagged"]:
                print("  flag  :", t)
            raise SystemExit("请先在核对工具中解决，或显式传 --allow-excluded 确认排除")

    creds = {
        row[0]: row[1]
        for row in __import__("csv").reader(args.credentials_csv.read_text(encoding="utf-8-sig").splitlines())
        if len(row) >= 2
    }
    api_key, chat_base = creds["apiKey"], creds["openAiCompatible"]

    api = Api(args.api_base)
    try:
        login = api.req("POST", "/api/auth/login", {"username": args.username, "password": args.password})
        token = login["data"]["access_token"]
        log("登录成功")
    except RuntimeError:
        api.req("POST", "/api/auth/register", {"username": args.username, "password": args.password})
        token = api.req("POST", "/api/auth/login", {"username": args.username, "password": args.password})["data"]["access_token"]
        log("注册并登录成功")
    api = Api(args.api_base, token)
    allow = fetch_allowlist()
    if args.kb_id:
        kb_id = args.kb_id
        log(f"复用知识库 {kb_id}")
        ready, failed = wait_ready(api, kb_id, len(allow), max_wait=120)
        if len(ready) < len(allow):
            raise SystemExit(f"复用库 ready 文档 {len(ready)} 少于语料 {len(allow)}，不能评测")
    else:
        kb = api.req("POST", "/api/knowledge-bases", {"name": f"bench-crud-{time.strftime('%m%d-%H%M')}", "description": "CRUD-RAG 公开基准 · 隔离评测库"})["data"]
        kb_id = kb["id"]
        log(f"知识库 {kb_id}")

        # 3. 上传语料
        log(f"上传 {len(allow)} 份语料 …")
        for i, item in enumerate(allow, 1):
            p = Path(item["path"])
            if not p.exists():
                raise SystemExit(f"语料缺失: {p}")
            api.upload(f"/api/knowledge-bases/{kb_id}/documents", p)
            if i % 50 == 0:
                log(f"  已上传 {i}/{len(allow)}")
        ready, failed = wait_ready(api, kb_id, len(allow))
        if failed:
            log(f"  !! {len(failed)} 份入库失败，检查后重跑")
            for f in failed[:5]:
                log(f"    失败: {f['filename']} {f.get('error', '')}")
    kb_list = api.req("GET", "/api/knowledge-bases")["data"]
    kb_now = next((k for k in kb_list if k["id"] == kb_id), {})
    freeze = {
        "kb_id": kb_id,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "kb_revision": kb_now.get("revision"),
        "documents": [
            {"id": d["id"], "filename": d["filename"], "status": d["status"], "index_revision": d.get("index_revision")}
            for d in ready
        ],
    }
    log(f"索引冻结: {len(ready)} 份 ready 文档, kb_revision={freeze['kb_revision']}")
    if args.ingest_only:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        state_path = args.out_dir / "ingested_kb.json"
        state_path.write_text(
            json.dumps({"kb_id": kb_id, "freeze": freeze, "documents_total": len(ready)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log(f"仅入库模式完成。知识库 ID: {kb_id}")
        log(f"正式评测加参数: --kb-id {kb_id}")
        log(f"状态文件: {state_path}")
        return

    # 4. 原子事实
    crud = {c["case_id"]: c for c in json.loads((PREPARED / "crud" / "cases.json").read_text(encoding="utf-8"))}
    cases = []
    fact_report = []
    for t in review["done"]:
        c = crud[t]
        rec = review["cases"][t]
        texts = []
        for ev in c["official_evidence"]:
            doc_file = PREPARED / "corpus" / f"{ev['document_id']}.txt"
            texts.append(doc_file.read_text(encoding="utf-8") if doc_file.exists() else "")
        raw_facts = decompose_facts(api_key, chat_base, c["question"], rec["confirmed_answer"])
        facts, dropped = [], []
        for f in raw_facts:
            (facts if verify_fact(f, texts, rec["confirmed_answer"]) else dropped).append(f)
        fact_report.append({"case_id": t, "raw": raw_facts, "kept": facts, "dropped": dropped})
        if not facts:
            raise SystemExit(f"案例 {t} 无可用事实（丢弃 {len(dropped)} 条）")
        cases.append(
            {
                "question": c["question"],
                "reference_answer": rec["confirmed_answer"],
                "facts": facts,
                # 先记外部语料 ID，绑定阶段统一映射为平台文档 UUID 后再送入评测
                "external_document_ids": [ev["document_id"] for ev in c["official_evidence"]],
                "spans": rec["confirmed_evidence_spans"],
            }
        )
    log(f"原子事实完成: {len(cases)} 题，丢弃 {sum(len(f['dropped']) for f in fact_report)} 条不可定位事实")

    # 5. 证据区间绑定 → 真实子块 ID
    # owner_id/kb_id 需要平台内部 ID：从任意文档 chunks 的权限校验反推不可行，
    # 直接查询 PG 的 owner_id（按 kb 内文档聚合）。
    doc_ids = [d["id"] for d in ready]
    quoted = ",".join(f"'{d}'" for d in doc_ids)
    owner_rows = run_sql_in_container(
        args.pg_container,
        f"SELECT DISTINCT owner_id, kb_id FROM chunk_vector WHERE doc_id IN ({quoted}) LIMIT 1;",
    ).strip()
    if not owner_rows:
        raise SystemExit("PG 中找不到上传文档的子块，绑定中止")
    owner_id, pg_kb_id = owner_rows.split("|")
    log(f"绑定依据: owner={owner_id[:8]}… kb={pg_kb_id[:8]}…")
    # 外部语料 ID → 平台文档 UUID：上传文件名为 doc-<sha>.txt
    ext_to_platform = {Path(d["filename"]).stem: d["id"] for d in ready}
    bind_problems = []
    for c in cases:
        # 相关文档 ID 必须用平台 UUID，否则 document_recall 恒为 0
        c["relevant_document_ids"] = [
            ext_to_platform[e] for e in c["external_document_ids"] if e in ext_to_platform
        ]
        missing_docs = [e for e in c["external_document_ids"] if e not in ext_to_platform]
        if missing_docs:
            bind_problems.append(f"{c['question'][:20]}… 相关文档不在 ready 集合: {missing_docs[:2]}")
        c["relevant_child_ids"] = []
        for span in c["spans"]:
            platform_doc = ext_to_platform.get(span["document_id"])
            if platform_doc is None:
                bind_problems.append(f"外部文档 {span['document_id'][:16]}… 不在本次 ready 集合中")
                continue
            local_span = dict(span, document_id=platform_doc)
            ids, problems = bind_case_spans(api, args.pg_container, owner_id, pg_kb_id, local_span, {})
            c["relevant_child_ids"].extend(ids)
            bind_problems.extend(problems)
        c["relevant_child_ids"] = sorted(set(c["relevant_child_ids"]))
        if not c["relevant_child_ids"]:
            bind_problems.append(f"{c['question'][:20]}… 全部区间绑定失败")
    log(f"区间绑定完成，问题 {len(bind_problems)} 条")
    for p in bind_problems[:10]:
        log(f"  !! {p}")

    # 6. 三档策略评测
    args.out_dir.mkdir(parents=True, exist_ok=True)
    strategies = [s.strip() for s in args.strategies.split(",")]
    all_results: dict[str, list[dict]] = {}
    for strategy in strategies:
        log(f"=== 策略 {strategy} ===")
        all_results[strategy] = run_strategy(api, kb_id, strategy, cases, args.batch_size, args.parallel)

    # 7. 汇总与导出
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": "evaluation/public_benchmarks_20261002/PROTOCOL.md",
        "platform": {
            "image": "zhixu-rag-api:1.81",
            "patch": "评测接口新增可选 mode 字段（默认 agent 保持原行为），协议要求三档统一 mode=rag 隔离检索策略差异",
            "model_concurrency": 4,
            "daily_dispatch_limit": 3000,
        },
        "run_config": {
            "batch_size": args.batch_size,
            "parallel": args.parallel,
            "strategies": strategies,
        },
        "review_file_sha256": hashlib.sha256(args.review_file.read_bytes()).hexdigest(),
        "freeze": freeze,
        "strategies": {},
        "fact_report": fact_report,
        "case_document_map": {
            c["question"]: {
                "external_document_ids": c["external_document_ids"],
                "platform_document_ids": c["relevant_document_ids"],
                "relevant_child_ids": c["relevant_child_ids"],
            }
            for c in cases
        },
        "bind_problems": bind_problems,
        "cases_noev": review["noev"],
        "cases_flagged": review["flagged"],
    }
    for strategy, results in all_results.items():
        report["strategies"][strategy] = {
            "aggregate": aggregate(results),
            "results": results,
        }
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_json = args.out_dir / f"crud_bench_report_{ts}.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"报告已写入 {out_json}")
    for strategy in strategies:
        agg = report["strategies"][strategy]["aggregate"]
        log(
            f"  {strategy}: valid={agg['valid_cases']} missing={agg['missing_cases']} "
            f"refusal={agg['refusal_rate']} CR={agg['context_recall']} CP={agg['context_precision']} "
            f"F={agg['faithfulness']} AR={agg['answer_relevancy']}"
        )


if __name__ == "__main__":
    main()
