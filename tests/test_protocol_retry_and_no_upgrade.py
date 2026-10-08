"""严格协议重试与"答案不得比原文更绝对"判定的回归测试。

对应两处修复：
1. check/grade 的结构校验失败时带纠正指令有限重试。生成温度为 0，
   相同提示词必然复现同一次失败，因此重试必须改变指令才有意义。
2. 范围核验原本要求答案与原文的语气/归属类别集合完全相等，
   会把"答案比原文更保守"也判为不通过；开关启用时只禁止升级。
"""

import json

import httpx
import pytest

from app.providers import CheckProtocolError, DashScopeProvider
from app.schemas import Judgment, _no_upgrade_preserved
from app.settings import Settings


def remote_settings(**overrides):
    return Settings(
        _env_file=None,
        model_provider="dashscope",
        dashscope_api_key="not-real",
        dashscope_http_base_url="https://example.test/api/v1",
        dashscope_chat_base_url="https://example.test/compatible-mode/v1",
        **overrides,
    )


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
    )


def test_no_upgrade_preserved_allows_more_cautious_answer():
    # 原文是断定/事实，答案更保守：允许
    assert _no_upgrade_preserved(["suggestion"], ["asserted"]) is True
    assert _no_upgrade_preserved(["suggestion"], ["fact"]) is True
    # 类别一致：允许
    assert _no_upgrade_preserved(["fact"], ["fact"]) is True
    assert _no_upgrade_preserved(["suggestion"], ["suggestion"]) is True


def test_no_upgrade_preserved_rejects_upgrade_and_undetermined():
    # 原文只是建议/预测，答案说成断定或事实：拒绝（核心防护不变）
    assert _no_upgrade_preserved(["asserted"], ["suggestion"]) is False
    assert _no_upgrade_preserved(["fact"], ["suggestion"]) is False
    assert _no_upgrade_preserved(["fact"], ["future"]) is False
    # 待定仍按原严格规则处理
    assert _no_upgrade_preserved(["undetermined"], ["fact"]) is False
    assert _no_upgrade_preserved(["suggestion"], ["undetermined"]) is False
    # 答案同时声称事实与建议，而原文没有事实：拒绝
    assert _no_upgrade_preserved(["fact", "suggestion"], ["suggestion"]) is False


async def test_check_protocol_failure_retries_with_changed_instruction(monkeypatch):
    """第一次结构失败、第二次成功；断言重试发生且纠正指令确实改变。"""
    corrections = []

    async def fake_once(self, schema, task, payload, correction=""):
        corrections.append(correction)
        if len(corrections) == 1:
            raise CheckProtocolError("check 未返回符合约束的 JSON")
        return Judgment(passed=True, reason="核验通过")

    monkeypatch.setattr(DashScopeProvider, "_structured_once", fake_once)
    provider = DashScopeProvider(
        remote_settings(protocol_retry_limit=1), httpx.MockTransport(lambda r: httpx.Response(200))
    )
    try:
        judgment = await provider.structured(Judgment, "check", {"answer": "答案", "sources": []})
        assert judgment.passed is True
        assert len(corrections) == 2, "应为初次 + 1 次重试"
        assert corrections[0] == "", "首次尝试不得改变原有指令"
        assert corrections[1] and "结构校验" in corrections[1], (
            "重试必须带纠正指令，否则温度 0 会复现同一次失败"
        )
    finally:
        await provider.close()


async def test_grade_protocol_failure_also_retries(monkeypatch):
    """grade 同样属于"结构失败即拒绝回答"，应享受同一重试策略。"""
    corrections = []

    async def fake_once(self, schema, task, payload, correction=""):
        corrections.append(correction)
        if len(corrections) == 1:
            raise CheckProtocolError("grade 未返回符合约束的 JSON")
        return Judgment(passed=True, reason="ok")

    monkeypatch.setattr(DashScopeProvider, "_structured_once", fake_once)
    provider = DashScopeProvider(
        remote_settings(protocol_retry_limit=1), httpx.MockTransport(lambda r: httpx.Response(200))
    )
    try:
        await provider.structured(Judgment, "grade", {"query": "q", "sources": []})
        assert len(corrections) == 2
    finally:
        await provider.close()


async def test_retry_limit_zero_keeps_original_behavior(monkeypatch):
    """protocol_retry_limit=0 时退化为修复前的行为：不重试。"""
    corrections = []

    async def fake_once(self, schema, task, payload, correction=""):
        corrections.append(correction)
        raise CheckProtocolError("invalid")

    monkeypatch.setattr(DashScopeProvider, "_structured_once", fake_once)
    provider = DashScopeProvider(
        remote_settings(protocol_retry_limit=0), httpx.MockTransport(lambda r: httpx.Response(200))
    )
    try:
        with pytest.raises(CheckProtocolError):
            await provider.structured(Judgment, "check", {"answer": "a", "sources": []})
        assert len(corrections) == 1
    finally:
        await provider.close()


async def test_check_protocol_failure_raises_after_retries_exhausted():
    """始终返回非法结构时，重试次数用尽后仍抛出 CheckProtocolError。"""
    calls = []

    async def handle(request):
        calls.append(1)
        return chat_response(json.dumps({"unexpected": "shape"}))

    provider = DashScopeProvider(
        remote_settings(protocol_retry_limit=1), httpx.MockTransport(handle)
    )
    try:
        with pytest.raises(CheckProtocolError):
            await provider.structured(Judgment, "check", {"answer": "答案", "sources": []})
        assert len(calls) == 2, "应为初次 + 1 次重试"
    finally:
        await provider.close()


async def test_non_strict_task_does_not_retry():
    """非 check/grade 任务保持原行为：解析失败不重试。"""
    calls = []

    async def handle(request):
        calls.append(1)
        return chat_response(json.dumps({"unexpected": "shape"}))

    provider = DashScopeProvider(
        remote_settings(protocol_retry_limit=1), httpx.MockTransport(handle)
    )
    try:
        with pytest.raises(Exception):
            await provider.structured(Judgment, "rewrite", {"q": "x"})
        assert len(calls) == 1
    finally:
        await provider.close()


def test_scope_no_upgrade_relaxation_defaults_on():
    """默认开启：三档实测拒答率全面下降且零回归；关闭后回到严格相等规则。"""
    assert Settings(_env_file=None).scope_no_upgrade_relaxation is True
    assert (
        Settings(_env_file=None, scope_no_upgrade_relaxation=False).scope_no_upgrade_relaxation
        is False
    )


async def test_default_relaxation_is_applied_on_the_atomic_check_path():
    """端到端确认默认值真的作用在核验链上，而不只是函数级行为。

    替身 Provider 没有 settings 时走保守回退，所以这里显式挂上 Settings；
    part_coverage_retry_limit 置 0 以免触发该替身未实现的 correction 形参。
    """
    from tests.test_predicate_roles import RoleProvider, review

    answer, content = "资料范围内，实际断言。[S1]", "实际断言。"

    # 答案=permission、原文=asserted：答案更保守，默认（放宽）应通过
    relaxed = RoleProvider(mode="permission", source_mode="asserted")
    relaxed.settings = Settings(_env_file=None, part_coverage_retry_limit=0)
    result, _ = await review(relaxed, answer, content)
    assert result["modality_preserved"] is True

    # 显式关回严格相等规则，同一输入应被拒
    strict = RoleProvider(mode="permission", source_mode="asserted")
    strict.settings = Settings(
        _env_file=None, part_coverage_retry_limit=0, scope_no_upgrade_relaxation=False
    )
    result, _ = await review(strict, answer, content)
    assert result["modality_preserved"] is False

    # 核心防护不变：原文只是建议，答案说成断定，两种配置都必须拒
    for flag in (True, False):
        provider = RoleProvider(mode="asserted", source_mode="suggestion")
        provider.settings = Settings(
            _env_file=None, part_coverage_retry_limit=0, scope_no_upgrade_relaxation=flag
        )
        result, _ = await review(provider, answer, content)
        assert result["modality_preserved"] is False


async def test_literal_error_retry_names_the_allowed_values():
    """枚举误用（例如把 voice 的 opinion 填进 mode）要被告知具体允许取值。"""
    from app.providers import validation_correction
    from app.source_parts import SourcePredicateParts

    try:
        SourcePredicateParts.model_validate(
            {
                "source_id": "S1",
                "source_span_id": "S1:E1",
                "predicates": [
                    {
                        "predicate_id": "P1",
                        "fragment_ids": ["S1:E1:F1"],
                        "mode": "opinion",
                        "voice": "opinion",
                        "reason": "主观评价",
                    }
                ],
            }
        )
    except ValueError as exc:
        correction = validation_correction(exc)
    else:  # pragma: no cover - 模型必须拒绝 opinion 作为 mode
        raise AssertionError("opinion 不应是合法 mode")
    assert "predicates.0.mode" in correction
    # 列出的是该字段的合法取值；不替模型选定其中一个，也不改写返回值。
    assert "'asserted'" in correction and "'suggestion'" in correction
    assert "重新分类" in correction
    assert "应填" not in correction and "改为 asserted" not in correction


async def test_specific_correction_replaces_generic_only_after_a_failure(monkeypatch):
    corrections = []

    async def fake_once(self, schema, task, payload, correction=""):
        corrections.append(correction)
        if len(corrections) == 1:
            error = CheckProtocolError("check 未返回符合约束的 JSON")
            error.correction = " 本次具体不合规之处：predicates.0.mode 只能是 'asserted'。"
            raise error
        return Judgment(passed=True, reason="核验通过")

    monkeypatch.setattr(DashScopeProvider, "_structured_once", fake_once)
    provider = DashScopeProvider(
        remote_settings(protocol_retry_limit=1), httpx.MockTransport(lambda r: httpx.Response(200))
    )
    try:
        await provider.structured(Judgment, "check", {"answer": "答案", "sources": []})
        assert corrections[0] == ""
        assert "predicates.0.mode" in corrections[1]
        assert "结构校验" in corrections[1]
    finally:
        await provider.close()


def test_relaxed_rule_permits_any_non_asserted_category_not_only_more_cautious():
    """固化放宽规则的实际边界——它比"只允许更保守"宽。

    实现只拒绝「答案含 asserted/fact 而原文没有」以及任一侧 undetermined；
    其余组合一律放行。因此 document_limitation 对 asserted（"资料未规定X"而原文
    断言了X）这类并非"更保守"而是相反的转换也被允许。

    这些格子目前只有语义核验（verdict 必须 supported）兜底，没有针对性反例验证。
    若将来收紧规则，本用例会失败并提醒同步更新记录。
    """
    # 更保守：允许（设计意图之内）
    assert _no_upgrade_preserved(["suggestion"], ["asserted"]) is True
    assert _no_upgrade_preserved(["planned"], ["asserted"]) is True
    # 并非更保守，而是相反或不同类别：当前同样放行
    assert _no_upgrade_preserved(["document_limitation"], ["asserted"]) is True
    assert _no_upgrade_preserved(["permission"], ["planned"]) is True
    assert _no_upgrade_preserved(["discussion"], ["asserted"]) is True
    # 唯一被拒的方向：弱证据说成强断言，以及待定
    assert _no_upgrade_preserved(["asserted"], ["suggestion"]) is False
    assert _no_upgrade_preserved(["fact"], ["planned"]) is False
    assert _no_upgrade_preserved(["suggestion"], ["undetermined"]) is False
