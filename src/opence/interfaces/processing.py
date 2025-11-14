"""Interfaces for processing raw documents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .data_models import Document, LLMRequest


class IProcessor(ABC):
    """Transforms acquired documents into analysis-ready snippets."""

    @abstractmethod
    def process(self, documents: List[Document], request: LLMRequest) -> List[Document]:
        """Return processed documents."""

