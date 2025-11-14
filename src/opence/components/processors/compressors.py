"""Lightweight compression utilities."""

from __future__ import annotations

from typing import List

from ...interfaces import Document, IProcessor, LLMRequest


class SimpleTruncationProcessor(IProcessor):
    """Keeps each document under a configurable character budget."""

    def __init__(self, max_chars: int = 2000) -> None:
        self.max_chars = max_chars

    def process(self, documents: List[Document], request: LLMRequest) -> List[Document]:
        trimmed: List[Document] = []
        for doc in documents:
            text = doc.content
            if len(text) > self.max_chars:
                text = text[: self.max_chars - 3] + "..."
            trimmed.append(
                Document(
                    id=doc.id,
                    content=text,
                    metadata={**doc.metadata, "compressed": len(doc.content) > self.max_chars},
                    score=doc.score,
                )
            )
        return trimmed
