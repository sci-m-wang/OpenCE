"""Reference constructor that turns documents into prompt context."""

from __future__ import annotations

from typing import List

from ...interfaces import ContextBundle, Document, IConstructor, LLMRequest


class FewShotConstructor(IConstructor):
    """Selects top-k documents and formats them as prompt references."""

    def __init__(self, top_k: int = 3, instructions: str | None = None) -> None:
        self.top_k = top_k
        self.instructions = instructions or (
            "Use the following knowledge snippets as authoritative references."
        )

    def construct(self, documents: List[Document], request: LLMRequest) -> ContextBundle:
        selected = documents[: self.top_k]
        return ContextBundle(
            instructions=self.instructions,
            references=selected,
            metadata={"question": request.question},
        )
