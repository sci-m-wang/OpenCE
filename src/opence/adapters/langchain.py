"""Adapters for interoperating with LangChain ecosystems."""

from __future__ import annotations

from typing import Any, List

from ..interfaces import Document, IAcquirer, LLMRequest


class LangChainRetrieverAcquirer(IAcquirer):
    """Wraps any LangChain retriever instance as an `IAcquirer`."""

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def acquire(self, request: LLMRequest) -> List[Document]:
        results = self.retriever.get_relevant_documents(request.question)
        documents: List[Document] = []
        for idx, doc in enumerate(results):
            page_content = getattr(doc, "page_content", "")
            metadata = getattr(doc, "metadata", {}) or {}
            documents.append(
                Document(
                    id=metadata.get("id", f"lc-{idx:04d}"),
                    content=page_content,
                    metadata=metadata,
                    score=getattr(doc, "score", None),
                )
            )
        return documents
