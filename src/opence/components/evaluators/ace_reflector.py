"""IEvaluator implementation backed by the ACE Reflector."""

from __future__ import annotations

import json
from typing import Any, Dict

from ...methods.ace.playbook import Playbook
from ...methods.ace.roles import GeneratorOutput, Reflector, ReflectorOutput
from ...interfaces import (
    ContextBundle,
    EvaluationSignal,
    IEvaluator,
    LLMRequest,
    ModelResponse,
)


class ACEReflectorEvaluator(IEvaluator):
    """Wraps the ACE Reflector into the IEvaluator contract."""

    def __init__(self, reflector: Reflector, playbook: Playbook) -> None:
        self.reflector = reflector
        self.playbook = playbook

    def evaluate(
        self,
        request: LLMRequest,
        response: ModelResponse,
        context: ContextBundle,
    ) -> EvaluationSignal:
        generator_output = self._parse_generator_output(response)
        reflection = self.reflector.reflect(
            question=request.question,
            generator_output=generator_output,
            playbook=self.playbook,
            ground_truth=context.metadata.get("ground_truth"),
            feedback=context.metadata.get("feedback") or response.metadata.get("feedback"),
        )
        feedback = reflection.key_insight or reflection.error_identification
        verdict = "ok" if reflection.key_insight else "needs_improvement"
        return EvaluationSignal(
            score=1.0 if reflection.key_insight else 0.0,
            verdict=verdict,
            feedback=feedback or "",
            metadata={
                "reflection_output": reflection,
                "reflection_raw": reflection.raw,
                "generator_output": generator_output,
            },
        )

    def _parse_generator_output(self, response: ModelResponse) -> GeneratorOutput:
        if "generator_output" in response.metadata:
            data = response.metadata["generator_output"]
            if isinstance(data, GeneratorOutput):
                return data
            if isinstance(data, dict):
                return self._generator_output_from_dict(data)
        payload = json.loads(response.text)
        if not isinstance(payload, dict):
            raise ValueError("LLM response must be a JSON object for ACE evaluation.")
        return self._generator_output_from_dict(payload)

    @staticmethod
    def _generator_output_from_dict(data: Dict[str, Any]) -> GeneratorOutput:
        reasoning = str(data.get("reasoning", ""))
        final_answer = str(data.get("final_answer", ""))
        bullet_ids = [str(item) for item in data.get("bullet_ids", []) if isinstance(item, (str, int))]
        return GeneratorOutput(
            reasoning=reasoning,
            final_answer=final_answer,
            bullet_ids=bullet_ids,
            raw=data,
        )
