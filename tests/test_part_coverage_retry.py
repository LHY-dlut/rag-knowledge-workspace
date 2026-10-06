"""逐谓词片段投影漏选实质片段时的定向重问。

只重述被漏掉的真实片段并要求重新归类；不自动补全、不改写类别、不放宽覆盖要求。
"""

import pytest

from app.answer_parts import FragmentCoverageError, bind_answer_parts
from app.answer_ranges import answer_fragments
from app.atomic_scope import project_parts
from app.check_protocol import answer_spans
from app.providers import CheckProtocolError
from app.schemas import AnswerRangePredicates
from tests.test_predicate_ranges import selection

ANSWER = "甲规定了期限，乙未规定期限。[S1]"


def focus():
    return answer_spans(ANSWER)[0]


def decision(ids, span="A:E1"):
    return AnswerRangePredicates(answer_span_id=span, predicates=[selection(ids)])


def bind(value):
    return bind_answer_parts(value, focus(), ANSWER)


def test_incomplete_selection_reports_the_missing_fragments():
    fragments = answer_fragments(focus(), ANSWER)
    with pytest.raises(FragmentCoverageError) as raised:
        bind(decision([fragments[0]["fragment_id"]]))
    missing = raised.value.missing
    assert [row["fragment_id"] for row in missing] == [fragments[1]["fragment_id"]]
    assert missing[0]["quote"] == fragments[1]["quote"]
    assert fragments[1]["fragment_id"] in raised.value.correction
    assert missing[0]["quote"] in raised.value.correction


class CoverageProvider:
    """第一次漏选第二个片段，带纠正指令重问时补全。"""

    supports_check_scope_binding = True
    supports_check_atomic_scope = True

    def __init__(self, settings, complete_on_retry=True):
        self.settings = settings
        self.complete_on_retry = complete_on_retry
        self.calls = []

    async def structured(self, schema, task, payload, correction=""):
        self.calls.append(correction)
        fragments = answer_fragments(focus(), ANSWER)
        ids = [fragments[0]["fragment_id"]]
        if correction and self.complete_on_retry:
            ids = [fragment["fragment_id"] for fragment in fragments]
        return decision(ids)


class Settings:
    def __init__(self, limit):
        self.part_coverage_retry_limit = limit


async def test_retry_names_the_missing_fragment_and_never_repairs_it():
    provider = CoverageProvider(Settings(1))
    independent, atoms = await project_parts(
        provider,
        AnswerRangePredicates,
        "check_answer_predicate_parts",
        {"answer": ANSWER},
        bind,
    )
    assert len(provider.calls) == 2
    assert provider.calls[0] == "", "首次尝试不得改变原有指令"
    assert "A:E1:F2" in provider.calls[1]
    assert "乙未规定期限" in provider.calls[1]
    assert independent.answer_span_id == "A:E1"
    assert [len(atom["answer_parts"]) for atom in atoms] == [2]


async def test_exhausted_retries_still_fail_closed():
    provider = CoverageProvider(Settings(1), complete_on_retry=False)
    with pytest.raises(FragmentCoverageError):
        await project_parts(
            provider,
            AnswerRangePredicates,
            "check_answer_predicate_parts",
            {"answer": ANSWER},
            bind,
        )
    assert len(provider.calls) == 2


async def test_zero_limit_keeps_the_original_single_call():
    provider = CoverageProvider(Settings(0))
    with pytest.raises(CheckProtocolError):
        await project_parts(
            provider,
            AnswerRangePredicates,
            "check_answer_predicate_parts",
            {"answer": ANSWER},
            bind,
        )
    assert provider.calls == [""]
