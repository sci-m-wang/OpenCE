import json
from dataclasses import dataclass
from typing import List

from opence import DummyLLMClient
from opence.core import ClosedLoopOrchestrator
from opence.models import DummyModelProvider
from opence.interfaces import (
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


class InMemoryAcquirer(IAcquirer):
    def __init__(self, documents: List[Document]) -> None:
        self._documents = documents

    def acquire(self, request: LLMRequest) -> List[Document]:
        return list(self._documents)


class PassthroughProcessor(IProcessor):
    def process(self, documents: List[Document], request: LLMRequest) -> List[Document]:
        return documents


class StaticConstructor(IConstructor):
    def construct(self, documents: List[Document], request: LLMRequest) -> ContextBundle:
        return ContextBundle(instructions="use the docs", references=documents, metadata={"question": request.question})


class EchoEvaluator(IEvaluator):
    def evaluate(self, request: LLMRequest, response: ModelResponse, context: ContextBundle) -> EvaluationSignal:
        return EvaluationSignal(score=1.0, verdict="ok", feedback=response.text)


class RecordEvolver(IEvolver):
    def evolve(self, context: ContextBundle, signal: EvaluationSignal) -> EvolutionDecision:
        return EvolutionDecision(summary="noop", updates={"feedback": signal.feedback})


def test_orchestrator_runs_closed_loop() -> None:
    documents = [Document(id="1", content="answer is 42")]
    acquirer = InMemoryAcquirer(documents)
    processors = [PassthroughProcessor()]
    constructor = StaticConstructor()
    evaluator = EchoEvaluator()
    evolver = RecordEvolver()

    client = DummyLLMClient()
    client.queue(json.dumps({"answer": "42"}))

    orchestrator = ClosedLoopOrchestrator(
        llm=client,
        acquirer=acquirer,
        processors=processors,
        constructor=constructor,
        evaluator=evaluator,
        evolver=evolver,
    )

    request = LLMRequest(question="What is the ultimate answer?")
    result = orchestrator.run(request)

    assert result.response.text.startswith("{")
    assert result.evaluation.verdict == "ok"
    assert result.evolution.summary == "noop"
    assert result.context.references[0].content == "answer is 42"


def test_orchestrator_accepts_model_provider() -> None:
    documents = [Document(id="1", content="answer is 42")]
    acquirer = InMemoryAcquirer(documents)
    constructor = StaticConstructor()
    provider = DummyModelProvider(responses=[json.dumps({"answer": "42"})])

    orchestrator = ClosedLoopOrchestrator(
        llm=provider,
        acquirer=acquirer,
        processors=[],
        constructor=constructor,
        evaluator=EchoEvaluator(),
        evolver=RecordEvolver(),
    )

    request = LLMRequest(question="test")
    result = orchestrator.run(request)

    assert result.evaluation.verdict == "ok"
