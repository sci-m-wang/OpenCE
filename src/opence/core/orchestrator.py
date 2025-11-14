"""Closed-loop orchestrator that drives the five-pillar pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ..interfaces import (
    ContextBundle,
    Document,
    EvaluationSignal,
    EvolutionDecision,
    IAcquirer,
    IConstructor,
    IEvaluator,
    IEvolver,
    IProcessor,
    LLMRequest,
    ModelResponse,
)
from ..models import BaseModelProvider, LLMClient


@dataclass
class LoopResult:
    """Outputs collected from a single orchestrator run."""

    request: LLMRequest
    prompt: str
    acquired_documents: List[Document]
    processed_documents: List[Document]
    context: ContextBundle
    response: ModelResponse
    evaluation: EvaluationSignal
    evolution: EvolutionDecision


class ClosedLoopOrchestrator:
    """Coordinates the five pillars to form a closed CE loop."""

    def __init__(
        self,
        *,
        llm: LLMClient | BaseModelProvider,
        acquirer: IAcquirer,
        processors: Sequence[IProcessor] | None,
        constructor: IConstructor,
        evaluator: IEvaluator,
        evolver: IEvolver,
    ) -> None:
        if isinstance(llm, BaseModelProvider):
            self.llm = llm.client()
        else:
            self.llm = llm
        self.acquirer = acquirer
        self.processors = list(processors or [])
        self.constructor = constructor
        self.evaluator = evaluator
        self.evolver = evolver

    def run(self, request: LLMRequest) -> LoopResult:
        documents = self.acquirer.acquire(request)
        processed = documents
        for processor in self.processors:
            processed = processor.process(processed, request)
        context_bundle = self.constructor.construct(processed, request)
        prompt = self._format_prompt(request, context_bundle)
        llm_response = self.llm.complete(prompt)
        response = ModelResponse(text=llm_response.text, metadata=llm_response.raw or {})
        evaluation = self.evaluator.evaluate(request, response, context_bundle)
        evolution = self.evolver.evolve(context_bundle, evaluation)
        return LoopResult(
            request=request,
            prompt=prompt,
            acquired_documents=documents,
            processed_documents=processed,
            context=context_bundle,
            response=response,
            evaluation=evaluation,
            evolution=evolution,
        )

    def _format_prompt(self, request: LLMRequest, context: ContextBundle) -> str:
        lines = [context.instructions or ""]
        for idx, doc in enumerate(context.references, start=1):
            lines.append(f"[ref-{idx}] {doc.content}")
        lines.append("Question: {question}".format(question=request.question))
        if request.context:
            lines.append(f"Additional context: {request.context}")
        return "\n\n".join(filter(None, lines))
