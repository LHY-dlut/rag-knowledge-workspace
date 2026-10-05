"""Deterministic artifacts from already checked statements; no new model facts."""

import hashlib
import html
import json
import math
import os
import re
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import ValidationError

from app.artifact_storage import PendingFile, artifact_directory, finish_io
from app.check_protocol import answer_citation_manifest, answer_spans, bind_check
from app.evidence_protocol import MARKER, match_quote
from app.models import Document, GeneratedArtifact
from app.schemas import CheckDecision

KINDS = {"chart", "report", "webpage"}
NUMBER = re.compile(
    r"(?<![\dA-Za-z.,+\-])(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(亿元|万元|千元|元|千米|公里|平方米|公斤|小时|kWh|吨|人|名|支|队|件|台|个|家|次|天|%|％)"
)


class ArtifactUnavailable(ValueError):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_facts(answer: str, sources: list[dict], check: dict, provider: str) -> list[dict]:
    if not isinstance(check, dict) or not check.get("passed") or not sources:
        raise ArtifactUnavailable("仅可从核验通过且有来源的答案生成成果")
    registry = answer_spans(answer)
    manifests = {r["answer_span_id"]: r for r in answer_citation_manifest(answer)}
    by_id = {s["source_id"]: s for s in sources}
    if len(by_id) != len(sources):
        raise ArtifactUnavailable("来源ID重复，不能生成成果")
    evidence = {}
    if provider == "dashscope":
        try:
            decision = CheckDecision.model_validate({"checks": check.get("checks", [])})
        except ValidationError:
            raise ArtifactUnavailable("核验记录缺少完整的事实与证据绑定") from None
        rebound = bind_check(decision, answer, sources)
        if not rebound.passed:
            raise ArtifactUnavailable("答案核验与当前来源不能完整绑定")
        evidence = {
            r["answer_binding"]["selected_span_id"]: r["evidence"]
            for r in rebound.evidence_protocol["claims"]
        }
    elif provider == "demo":
        # Demo only offers exact extraction. Never describe lexical checks as
        # a semantic model judgment or accept a paraphrase through similarity.
        for span in registry:
            body = MARKER.sub("", span["quote"]).removeprefix("资料中记录：").strip()
            matches = [
                match_quote(body, by_id.get(i))
                for i in manifests[span["span_id"]]["cited_source_ids"]
            ]
            if not body or not matches or any(not m["passed"] for m in matches):
                raise ArtifactUnavailable("Demo成果只支持逐字摘录，不能替代语义核验")
            evidence[span["span_id"]] = matches
    else:
        raise ArtifactUnavailable("未知核验模式")
    facts = []
    for span in registry:
        ids = manifests[span["span_id"]]["cited_source_ids"]
        if not ids or not set(ids) <= by_id.keys() or not evidence.get(span["span_id"]):
            raise ArtifactUnavailable("存在无依据的答案片段")
        facts.append(
            {
                "span_id": span["span_id"],
                "text": span["quote"],
                "answer_start": span["source_start"],
                "answer_end": span["source_end"],
                "source_ids": ids,
                "evidence": evidence[span["span_id"]],
            }
        )
    return facts


def numeric_charts(facts: list[dict]) -> tuple[list[dict], list[dict]]:
    groups, missing = {}, []
    for fact in facts:
        text = MARKER.sub("", fact["text"]).removeprefix("资料中记录：").strip()
        # A numeric token in a bounded denial is not an observed quantity.
        # Charts conservatively require an exact extracted statement, as well
        # as the existing semantic verdict. Reports retain checked paraphrases.
        if re.search(
            r"不|未|无|不能|无法|可能|假设|预计|约|至少|最多|超过|低于|高于|小于|大于", text
        ):
            missing.append({"span_id": fact["span_id"], "reason": "numeric_assertion_ambiguous"})
            continue
        if not any(
            match_quote(text, {"content": r["original_quote"]})["passed"] for r in fact["evidence"]
        ):
            missing.append(
                {"span_id": fact["span_id"], "reason": "exact_numeric_statement_required"}
            )
            continue
        matches = list(NUMBER.finditer(text))
        if len(matches) != 1:
            missing.append(
                {
                    "span_id": fact["span_id"],
                    "reason": "one_numeric_value_per_checked_statement_required",
                }
            )
            continue
        match = matches[0]
        raw, unit = match[0], match[2]
        value = float(Decimal(match[1].replace(",", "")))
        if not math.isfinite(value) or abs(value) > 1e12:
            missing.append({"span_id": fact["span_id"], "reason": "numeric_range_unsupported"})
            continue
        positions = []
        for ref in fact["evidence"]:
            quote = ref["original_quote"]
            offset = quote.find(raw)
            if offset >= 0:
                positions.append(
                    {
                        "source_id": ref["source_id"],
                        "document_id": ref["document_id"],
                        "parent_id": ref.get("parent_id"),
                        "source_start": ref["source_start"] + offset,
                        "source_end": ref["source_start"] + offset + len(raw),
                        "original_numeric_quote": raw,
                    }
                )
        if not positions:
            # e.g. 'cannot conclude 7 days' with evidence only saying 'not specified'.
            missing.append(
                {
                    "span_id": fact["span_id"],
                    "reason": "numeric_token_missing_from_checked_evidence",
                }
            )
            continue
        label = (text[: match.start()] + "「数量」" + text[match.end() :]).strip()
        groups.setdefault(unit, []).append(
            {
                "label": label,
                "value": value,
                "raw_value": match[1],
                "unit": unit,
                "fact_span_id": fact["span_id"],
                "statement": fact["text"],
                "source_ids": fact["source_ids"],
                "numeric_positions": positions,
            }
        )
    return [{"unit": unit, "points": points[:30]} for unit, points in groups.items()], missing


def build_payload(
    kind: str, query: str, answer: str, sources: list[dict], check: dict, provider: str
) -> dict:
    if kind not in KINDS:
        raise ArtifactUnavailable("成果类型不支持")
    facts = checked_facts(answer, sources, check, provider)
    charts, missing = numeric_charts(facts) if kind == "chart" else ([], [])
    nodes = [
        {
            "id": "source:" + s["source_id"],
            "name": f"[{s['source_id']}] {s['filename']}",
            "category": "source",
        }
        for s in sources
    ]
    nodes += [
        {"id": "fact:" + f["span_id"], "name": f["text"], "category": "checked_statement"}
        for f in facts
    ]
    edges = [
        {"source": "source:" + sid, "target": "fact:" + f["span_id"], "relation": "cited_evidence"}
        for f in facts
        for sid in f["source_ids"]
    ]
    return {
        "schema_version": 1,
        "type": kind,
        "title": query,
        "answer": answer,
        "answer_sha256": digest(answer.encode()),
        "verification": "real_model_checked"
        if provider == "dashscope"
        else "demo_exact_extraction_only",
        "facts": facts,
        "citations": sources,
        "charts": charts,
        "chart_kind": "numeric" if charts else "evidence_graph",
        "numeric_unavailable": missing,
        "graph": {"nodes": nodes, "edges": edges},
        "calculation": {"performed": False, "totals_growth_or_sorting_fabricated": False},
    }


def render_html(payload: dict) -> str:
    esc = html.escape
    paragraphs = "".join(f"<p>{esc(f['text'])}</p>" for f in payload["facts"])
    references = "".join(
        f"<section><h3>[{esc(s['source_id'])}] {esc(s['filename'])}</h3><p>{esc(s['location'])}</p>"
        f"<p>文档版本：{s['document_revision']}</p><pre>{esc(s['content'])}</pre></section>"
        for s in payload["citations"]
    )
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">"
        f"<title>{esc(payload['title'])}</title><style>body{{font-family:system-ui;color:#173042;background:#f4f7fa;margin:0}}"
        "main{max-width:1000px;margin:24px auto;padding:32px;background:white;border-radius:16px}p,pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.8}"
        "section{border-top:1px solid #dae3ea;padding:16px 0}h1{font-size:26px}small{color:#566a77}</style></head><body>"
        f"<main><h1>{esc(payload['title'])}</h1><small>基于核验答案及保存的证据版本 · {esc(payload['verification'])}</small>"
        f"<h2>已核验内容</h2>{paragraphs}<h2>引用与证据</h2>{references}</main></body></html>"
    )


def file_path(settings, artifact_id: str, kind: str) -> Path:
    UUID(artifact_id)
    if kind not in KINDS:
        raise ValueError("Invalid artifact kind")
    base = artifact_directory(settings)
    path = (base / (artifact_id + (".json" if kind == "chart" else ".html"))).resolve()
    if not path.is_relative_to(base):
        raise ValueError("Invalid artifact file path")
    return path


async def persist_artifact(run, kind, payload, settings, connection, pending_files):
    old = (
        await GeneratedArtifact.filter(
            run_id=run.id, kind=kind, answer_sha256=payload["answer_sha256"]
        )
        .using_db(connection)
        .first()
    )
    if old:
        return old, False
    artifact_id = str(uuid4())
    path = file_path(settings, artifact_id, kind)
    data = (
        json.dumps(payload, ensure_ascii=False, indent=2)
        if kind == "chart"
        else render_html(payload)
    ).encode()
    if len(data) > 2_000_000:
        raise ArtifactUnavailable("成果超过大小上限")
    pending = PendingFile(artifact_id, path)
    pending_files.append(pending)

    def save_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            pending.owned = True
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    await finish_io(save_file)
    item = await GeneratedArtifact.create(
        id=artifact_id,
        owner_id=run.owner_id,
        kb_id=run.kb_id,
        run_id=run.id,
        kind=kind,
        answer_sha256=payload["answer_sha256"],
        payload=payload,
        storage_path=path.name,
        file_sha256=digest(data),
        using_db=connection,
    )
    return item, True


def artifact_dto(item, current: bool = True):
    return {
        "id": item.id,
        "run_id": item.run_id,
        "type": item.kind,
        "created_at": item.created_at.isoformat(),
        "payload": item.payload,
        "current_evidence": current,
        "download_url": f"/api/artifacts/{item.id}/download",
        "preview_html": render_html(item.payload) if item.kind != "chart" else None,
    }


async def list_artifacts(owner_id: str, run_id: str):
    items = await GeneratedArtifact.filter(owner_id=owner_id, run_id=run_id).order_by("created_at")
    document_ids = {s["document_id"] for item in items for s in item.payload["citations"]}
    docs = await Document.filter(owner_id=owner_id, id__in=sorted(document_ids))
    live = {d.id: d.index_revision for d in docs if d.status == "ready"}
    return [
        artifact_dto(
            item,
            all(
                live.get(s["document_id"]) == s["document_revision"]
                for s in item.payload["citations"]
            ),
        )
        for item in items
    ]
