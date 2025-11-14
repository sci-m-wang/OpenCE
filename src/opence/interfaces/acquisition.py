"""Interfaces for retrieving source knowledge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .data_models import Document, LLMRequest


class IAcquirer(ABC):
    """Fetches raw documents from arbitrary sources."""

    @abstractmethod
    def acquire(self, request: LLMRequest) -> List[Document]:
        """Return documents relevant to the request."""

