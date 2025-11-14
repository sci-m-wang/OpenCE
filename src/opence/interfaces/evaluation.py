"""Interfaces for closed-loop evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .data_models import ContextBundle, EvaluationSignal, LLMRequest, ModelResponse


class IEvaluator(ABC):
    """Produces quality signals from LLM responses."""

    @abstractmethod
    def evaluate(
        self,
        request: LLMRequest,
        response: ModelResponse,
        context: ContextBundle,
    ) -> EvaluationSignal:
        """Return evaluation feedback for the closed loop."""
