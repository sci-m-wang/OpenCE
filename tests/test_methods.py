import json
from dataclasses import dataclass
from typing import List

from opence import DummyLLMClient
from opence.interfaces import ContextBundle, Document, IAcquirer, IConstructor, LLMRequest
from opence.methods import ACEClosedLoopMethod


class InMemoryAcquirer(IAcquirer):
    def __init__(self, documents: List[Document]) -> None:
        self._documents = documents

    def acquire(self, request: LLMRequest) -> List[Document]:
        return list(self._documents)


class MetadataConstructor(IConstructor):
    def construct(self, documents, request: LLMRequest) -> ContextBundle:
        return ContextBundle(
            instructions="use context",
            references=documents,
            metadata={"question": request.question, "ground_truth": request.metadata.get("ground_truth")},
        )


def queue_reflector_client(client: DummyLLMClient) -> None:
    client.queue(
        json.dumps(
            {
                "reasoning": "analysis",
                "error_identification": "",
                "root_cause_analysis": "",
                "correct_approach": "",
                "key_insight": "answer is 42",
                "bullet_tags": [],
            }
        )
    )


def queue_curator_client(client: DummyLLMClient) -> None:
    client.queue(
        json.dumps(
            {
                "reasoning": "adding",
                "operations": [
                    {
                        "type": "ADD",
                        "section": "defaults",
                        "content": "Keep 42 handy.",
                        "metadata": {"helpful": 1},
                    }
                ],
            }
        )
    )


def queue_generator_client(client: DummyLLMClient) -> None:
    client.queue(
        json.dumps(
            {
                "reasoning": "found answer",
                "bullet_ids": [],
                "final_answer": "42",
            }
        )
    )


def test_ace_closed_loop_method_builds_orchestrator() -> None:
    docs = [Document(id="1", content="life is 42")]
    acquirer = InMemoryAcquirer(docs)
    constructor = MetadataConstructor()

    generator_client = DummyLLMClient()
    reflector_client = DummyLLMClient()
    curator_client = DummyLLMClient()

    queue_generator_client(generator_client)
    queue_reflector_client(reflector_client)
    queue_curator_client(curator_client)

    method = ACEClosedLoopMethod(
        generator_llm=generator_client,
        reflector_llm=reflector_client,
        curator_llm=curator_client,
        acquirer=acquirer,
        processors=[],
        constructor=constructor,
    )

    artifacts = method.build()
    result = artifacts.orchestrator.run(LLMRequest(question="?", metadata={"ground_truth": "42"}))

    assert result.evaluation.score == 1.0
    assert method.playbook.bullets()
