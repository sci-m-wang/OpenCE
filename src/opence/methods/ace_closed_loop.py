"""Reference method wiring the ACE components into a closed loop."""

from __future__ import annotations

from typing import Optional, Sequence

from .ace import Curator, Playbook, Reflector
from ..components import (
    ACECuratorEvolver,
    ACEReflectorEvaluator,
    FewShotConstructor,
    FileSystemAcquirer,
    KeywordBoostReranker,
    SimpleTruncationProcessor,
)
from ..interfaces import IAcquirer, IConstructor, IProcessor
from ..models import LLMClient
from .base import BaseMethod, PillarBundle


class ACEClosedLoopMethod(BaseMethod):
    """Ships ACE's evaluator + evolver as an out-of-the-box method."""

    name = "ace.closed_loop"
    description = "Closed-loop orchestration using ACE reflector + curator."

    def __init__(
        self,
        *,
        generator_llm: LLMClient,
        reflector_llm: LLMClient,
        curator_llm: LLMClient,
        playbook: Optional[Playbook] = None,
        acquirer: Optional[IAcquirer] = None,
        processors: Optional[Sequence[IProcessor]] = None,
        constructor: Optional[IConstructor] = None,
    ) -> None:
        self.playbook = playbook or Playbook()
        self.reflector = Reflector(reflector_llm)
        self.curator = Curator(curator_llm)

        evaluator = ACEReflectorEvaluator(self.reflector, self.playbook)
        evolver = ACECuratorEvolver(self.curator, self.playbook)

        default_acquirer = acquirer or FileSystemAcquirer("docs")
        default_processors: Sequence[IProcessor] = processors or (
            KeywordBoostReranker(["safety", "fire", "response"]),
            SimpleTruncationProcessor(),
        )
        default_constructor = constructor or FewShotConstructor()

        bundle = PillarBundle(
            acquirer=default_acquirer,
            processors=list(default_processors),
            constructor=default_constructor,
            evaluator=evaluator,
            evolver=evolver,
        )
        super().__init__(generator_llm, bundle)
