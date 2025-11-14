"""Very small heuristic rerankers."""

from __future__ import annotations

from typing import Iterable, List

from ...interfaces import Document, IProcessor, LLMRequest


class KeywordBoostReranker(IProcessor):
    """Boosts documents containing high-priority keywords."""

    def __init__(self, keywords: Iterable[str]) -> None:
        self.keywords = [kw.lower() for kw in keywords]

    def process(self, documents: List[Document], request: LLMRequest) -> List[Document]:
        def score(doc: Document) -> float:
            base = doc.score or 0.0
            text = doc.content.lower()
            boost = sum(text.count(kw) for kw in self.keywords)
            return base + boost

        return sorted(documents, key=score, reverse=True)
