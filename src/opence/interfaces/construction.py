"""Interfaces for building the final prompt/context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .data_models import ContextBundle, Document, LLMRequest


class IConstructor(ABC):
    """Synthesizes a context bundle from processed documents."""

    @abstractmethod
    def construct(self, documents: List[Document], request: LLMRequest) -> ContextBundle:
        """Return the prompt-ready context."""

