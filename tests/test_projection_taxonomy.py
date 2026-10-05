"""Wire regressions for the contradictory rules found in frozen V78 requests."""

import json

import httpx
import pytest

from app.answer_ranges import answer_fragments
from app.providers import DashScopeProvider
from app.schemas import AnswerRangePredicates
from app.settings import Settings
from app.source_windows import SourceWindowParts, source_windows, window_fragments


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["check_answer_predicate_parts", "check_source_predicate_window"])
async def test_independent_projections_share_one_property_rule_and_keep_original_positions(task):
    text = "顾问赵航预计预约将减少等待，这份判断属于预测。"
    captured = []
    source = {"source_id": "S7", "content": text}
    if task == "check_source_predicate_window":
        window = source_windows(source)[0]
        fragments = window_fragments(source, window)
        payload = {
            "source_id": "S7",
            "source_text": text,
            "focus_source_window": window,
            "source_fragments": fragments,
        }
        schema = SourceWindowParts
        identity = {"source_id": "S7", "source_window_id": window["window_id"]}
    else:
        focus = {"span_id": "A7", "source_start": 0, "source_end": len(text), "quote": text}
        fragments = answer_fragments(focus, text)
        payload = {
            "answer": text,
            "focus_answer_span_id": "A7",
            "focus_answer_span": focus,
            "answer_context_spans": [focus],
            "answer_fragments": fragments,
        }
        schema = AnswerRangePredicates
        identity = {"answer_span_id": "A7"}
    value = {
        **identity,
        "predicates": [
            {
                "predicate_id": "forecast",
                "fragment_ids": [fragments[0]["fragment_id"]],
                "mode": "future",
                "voice": "opinion",
                "reason": "Mock仅检查协议和未修改位置",
            },
            {
                "predicate_id": "property",
                "fragment_ids": [f["fragment_id"] for f in fragments[1:]],
                "mode": "asserted",
                "voice": "fact",
                "reason": "Mock仅检查独立性质关系",
            },
        ],
    }

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}}]})

    cfg = Settings(
        _env_file=None,
        check_source_window_projection=True,
        dashscope_chat_base_url="https://example.test/v1",
    )
    provider = DashScopeProvider(cfg, httpx.MockTransport(respond))
    try:
        parsed = await provider.structured(schema, task, payload)
    finally:
        await provider.close()
    assert parsed.model_dump() == value
    body = captured[0]
    system = body["messages"][0]["content"]
    sent = json.loads(body["messages"][1]["content"])
    assert system.count("两次投影只使用以下唯一分类表") == 1
    assert "不单独引入资料不足或额外事实类别" not in system
    assert "不要把同一个预测的性质说明另拆" not in system
    assert "对预测性质的asserted/fact描述，不证明预测已实现" in system
    assert not {"query", "grade", "score", "system_results"} & sent.keys()
    if task == "check_source_predicate_window":
        assert "answer" not in sent and sent["source_text"] == text
    assert sent.get("answer_fragments", sent.get("source_fragments")) == fragments
    for fragment in fragments:
        assert text[fragment["source_start"] : fragment["source_end"]] == fragment["quote"]
