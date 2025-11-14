"""Canonical data models shared across OpenCE interfaces."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Atomic unit of information passed between CE stages."""

    id: str = Field(default_factory=lambda: "doc-unknown")
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None


class ContextBundle(BaseModel):
    """Structured context that will be serialized into the LLM prompt."""

    instructions: str = ""
    references: List[Document] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMRequest(BaseModel):
    """Represents a problem to solve via LLM."""

    question: str
    context: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    """Container for raw LLM outputs."""

    text: str
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationSignal(BaseModel):
    """Evaluation feedback produced by an IEvaluator implementation."""

    score: float = 0.0
    verdict: str = "unknown"
    feedback: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvolutionDecision(BaseModel):
    """Action emitted by an IEvolver to update persistent strategies."""

    summary: str = ""
    updates: Dict[str, Any] = Field(default_factory=dict)
