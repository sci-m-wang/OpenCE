import json

from opence.methods.ace import Curator, Playbook, Reflector
from opence import DummyLLMClient
from opence.components.evaluators.ace_reflector import ACEReflectorEvaluator
from opence.components.evolvers.ace_curator import ACECuratorEvolver
from opence.interfaces import ContextBundle, LLMRequest, ModelResponse


def test_ace_wrappers_update_playbook() -> None:
    playbook = Playbook()

    reflector_client = DummyLLMClient()
    reflector_client.queue(
        json.dumps(
            {
                "reasoning": "analysis",
                "error_identification": "",
                "root_cause_analysis": "",
                "correct_approach": "",
                "key_insight": "store the answer",
                "bullet_tags": [],
            }
        )
    )
    reflector = Reflector(reflector_client)
    evaluator = ACEReflectorEvaluator(reflector, playbook)

    response = ModelResponse(
        text=json.dumps(
            {
                "reasoning": "worked",
                "bullet_ids": [],
                "final_answer": "42",
            }
        ),
        metadata={"feedback": "correct"},
    )
    request = LLMRequest(question="What is 42?")
    context = ContextBundle(metadata={"question": request.question, "ground_truth": "42"})

    signal = evaluator.evaluate(request, response, context)
    assert signal.score == 1.0
    assert "reflection_raw" in signal.metadata

    curator_client = DummyLLMClient()
    curator_client.queue(
        json.dumps(
            {
                "reasoning": "adding",
                "operations": [
                    {
                        "type": "ADD",
                        "section": "defaults",
                        "content": "Answer 42 when unsure.",
                        "metadata": {"helpful": 1},
                    }
                ],
            }
        )
    )
    curator = Curator(curator_client)
    evolver = ACECuratorEvolver(curator, playbook)

    decision = evolver.evolve(context, signal)
    assert "applied" in decision.summary
    assert playbook.bullets()
