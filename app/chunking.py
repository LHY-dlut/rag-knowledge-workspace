import re
from dataclasses import dataclass, field
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from app.schemas import RetrievalConfig


@dataclass
class ParsedUnit:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParentSlice:
    id: str
    ordinal: int
    content: str
    metadata: dict[str, Any]
    children: list[tuple[str, str, int, int]]


def split_spans(text: str, size: int, overlap: int = 0) -> list[tuple[int, int]]:
    """Character budget, exact offsets, prefer paragraphs/sentence boundaries.

    Chinese characters are NOT tokenizer tokens. Preserve the original substring,
    even if a very long paragraph forces a hard split. Every loop makes progress.
    """
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("Invalid span size/overlap")
    spans, start = [], 0
    boundaries = [m.end() for m in re.finditer(r"\n\s*\n|[。！？.!?]\s*|\n", text)]
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            choices = [p for p in boundaries if start + size // 2 <= p <= end]
            if choices:
                end = choices[-1]
        spans.append((start, end))
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return spans


def children_for_parent(parent_id: str, content: str, config: RetrievalConfig):
    if config.chunk_strategy != "parent_child":
        child_id = str(uuid5(NAMESPACE_URL, f"{parent_id}:0:{len(content)}:{content}"))
        return [(child_id, content, 0, len(content))]
    return [
        (
            str(uuid5(NAMESPACE_URL, f"{parent_id}:{start}:{end}:{content[start:end]}")),
            content[start:end],
            start,
            end,
        )
        for start, end in split_spans(content, config.child_size, config.child_overlap)
        if content[start:end].strip()
    ]


def parent_child_chunks(doc_id: str, units: list[ParsedUnit], config: RetrievalConfig):
    parents: list[ParentSlice] = []
    size, overlap = config.parent_size, 0
    if config.chunk_strategy == "recursive":
        size, overlap = config.recursive_size, config.recursive_overlap
    elif config.chunk_strategy == "recursive_short":
        size, overlap = config.short_size, config.short_overlap
    for unit_index, unit in enumerate(units):
        pattern = {
            "laws": r"(?m)^\s*第[一二三四五六七八九十百千万零〇\d]+[条章节]",
            "qa": r"(?m)^(?:问[：:]|问题\s*\d*[：:]|Q\s*\d*[：:])",
        }.get(config.structure_mode)
        boundaries = (
            sorted({0, len(unit.content), *(m.start() for m in re.finditer(pattern, unit.content))})
            if pattern
            else [0, len(unit.content)]
        )
        spans = [
            (base + a, base + b)
            for base, stop in zip(boundaries, boundaries[1:], strict=False)
            for a, b in split_spans(unit.content[base:stop], size, overlap)
        ]
        for start, end in spans:
            content = unit.content[start:end]
            if not content.strip():
                continue
            parent_id = str(uuid5(NAMESPACE_URL, f"{doc_id}:{unit_index}:{start}:{end}"))
            metadata = {
                **unit.metadata,
                "unit": unit_index,
                "start": start,
                "end": end,
                "chunk_strategy": config.chunk_strategy,
                "structure_mode": config.structure_mode,
            }
            parents.append(
                ParentSlice(
                    parent_id,
                    len(parents),
                    content,
                    metadata,
                    children_for_parent(parent_id, content, config),
                )
            )
    return parents
