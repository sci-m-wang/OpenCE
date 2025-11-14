"""IEvolver implementation using the ACE curator + playbook."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ...methods.ace.playbook import Playbook
from ...methods.ace.roles import BulletTag, Curator, ReflectorOutput
from ...interfaces import ContextBundle, EvaluationSignal, EvolutionDecision, IEvolver


class ACECuratorEvolver(IEvolver):
    """Applies ACE curator deltas as an evolution strategy."""

    def __init__(self, curator: Curator, playbook: Playbook) -> None:
        self.curator = curator
        self.playbook = playbook

    def evolve(self, context: ContextBundle, signal: EvaluationSignal) -> EvolutionDecision:
        reflection = self._resolve_reflection(signal.metadata)
        if reflection is None:
            return EvolutionDecision(summary="no-reflection", updates={})
        question_context = self._build_question_context(context, signal)
        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=question_context,
            progress=context.metadata.get("progress", "offline"),
        )
        self.playbook.apply_delta(curator_output.delta)
        return EvolutionDecision(
            summary=f"applied {len(curator_output.delta.operations)} operations",
            updates={
                "operations": [op.to_json() for op in curator_output.delta.operations],
                "curator_reasoning": curator_output.raw.get("reasoning"),
            },
        )

    def _resolve_reflection(self, metadata: Dict[str, Any]) -> Optional[ReflectorOutput]:
        payload = metadata.get("reflection_output")
        if isinstance(payload, ReflectorOutput):
            return payload
        raw = metadata.get("reflection_raw")
        if isinstance(raw, dict):
            return self._build_reflection_from_raw(raw)
        return None

    def _build_question_context(
        self,
        context: ContextBundle,
        signal: EvaluationSignal,
    ) -> str:
        generator_output = signal.metadata.get("generator_output")
        generator_json = json.dumps(getattr(generator_output, "raw", {}), ensure_ascii=False)
        lines = [
            f"question: {context.metadata.get('question', '')}",
            f"instructions: {context.instructions}",
            f"generator: {generator_json}",
        ]
        return "\n".join(lines)

    def _build_reflection_from_raw(self, raw: Dict[str, Any]) -> ReflectorOutput:
        bullet_tags = []
        for item in raw.get("bullet_tags", []) or []:
            if isinstance(item, dict) and "id" in item and "tag" in item:
                bullet_tags.append(BulletTag(id=str(item["id"]), tag=str(item["tag"])) )
        return ReflectorOutput(
            reasoning=str(raw.get("reasoning", "")),
            error_identification=str(raw.get("error_identification", "")),
            root_cause_analysis=str(raw.get("root_cause_analysis", "")),
            correct_approach=str(raw.get("correct_approach", "")),
            key_insight=str(raw.get("key_insight", "")),
            bullet_tags=bullet_tags,
            raw=raw,
        )
