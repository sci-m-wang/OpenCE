"""Foundations for defining composite Context Engineering methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from ..core import ClosedLoopOrchestrator
from ..interfaces import (
    IAcquirer,
    IConstructor,
    IEvaluator,
    IEvolver,
    IProcessor,
)
from ..models import LLMClient


@dataclass
class MethodArtifacts:
    """Container returned by every method factory."""

    orchestrator: ClosedLoopOrchestrator
    metadata: Dict[str, Any]


@dataclass
class PillarBundle:
    """Concrete components wired into a closed-loop method."""

    acquirer: IAcquirer
    processors: Sequence[IProcessor]
    constructor: IConstructor
    evaluator: IEvaluator
    evolver: IEvolver


class BaseMethod:
    """Common convenience for method authors."""

    name: str = "method"
    description: str = ""

    def __init__(self, llm: LLMClient, bundle: PillarBundle) -> None:
        self.llm = llm
        self.bundle = bundle

    def build(self) -> MethodArtifacts:
        orchestrator = ClosedLoopOrchestrator(
            llm=self.llm,
            acquirer=self.bundle.acquirer,
            processors=list(self.bundle.processors),
            constructor=self.bundle.constructor,
            evaluator=self.bundle.evaluator,
            evolver=self.bundle.evolver,
        )
        return MethodArtifacts(orchestrator=orchestrator, metadata={"name": self.name})


class MethodRegistry:
    """Simple registry so users can look up prebuilt methods by name."""

    def __init__(self) -> None:
        self._methods: Dict[str, BaseMethod] = {}

    def register(self, method: BaseMethod) -> None:
        key = method.name
        if key in self._methods:
            raise ValueError(f"Method '{key}' already registered")
        self._methods[key] = method

    def get(self, name: str) -> BaseMethod:
        if name not in self._methods:
            raise KeyError(f"Method '{name}' not found")
        return self._methods[name]

    def available(self) -> List[str]:
        return sorted(self._methods)
