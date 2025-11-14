"""Interfaces for strategy evolution."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .data_models import ContextBundle, EvaluationSignal, EvolutionDecision


class IEvolver(ABC):
    """Consumes evaluation signals and mutates long-term state."""

    @abstractmethod
    def evolve(
        self,
        context: ContextBundle,
        signal: EvaluationSignal,
    ) -> EvolutionDecision:
        """Return a decision capturing how state should update."""

