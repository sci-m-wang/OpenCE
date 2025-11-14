# OpenCE: Closed-Loop Context Engineering Toolkit

OpenCE is a **pluggable meta-framework** for building closed-loop Context Engineering (CE) systems. It evolves the original community ACE reproduction into a toolkit that can *sense → reason → evaluate → evolve* its own strategies.

## Why Closed-Loop CE?

Classical RAG stacks are open loops: they fetch context once and immediately respond. OpenCE adds two missing pillars:

1. **Evaluation** – automatically score every LLM response using domain-specific evaluators (ACE Reflector, RAGAS, etc.).
2. **Evolution** – feed those evaluation signals into long-term memory/strategy modules (ACE Curator, adaptive RAG policies, …).

This creates a self-improving flywheel where every new interaction strengthens future contexts.

## The Five Pillars Architecture

OpenCE standardizes five interfaces so that any CE system can be composed as Lego bricks:

| Pillar | Interface | Responsibility |
| --- | --- | --- |
| Acquisition | `IAcquirer` | Perception layer (databases, web, LangChain retrievers). |
| Processing | `IProcessor` | Cleans, deduplicates, compresses, or reranks acquired knowledge. |
| Construction | `IConstructor` | Builds the final prompt/context bundle (few-shot selection, dynamic instructions). |
| Evaluation | `IEvaluator` | Scores LLM responses; outputs rich feedback signals. |
| Evolution | `IEvolver` | Consumes evaluation signals to update long-term strategies (playbooks, memories, knobs). |

Each pillar is defined in `src/opence/interfaces/` (the **soul**), implemented natively in `src/opence/components/` (the **batteries**), and can be connected to external ecosystems through `src/opence/adapters/` (the **glue**).

## Repository Layout

```
src/
└── opence/
    ├── interfaces/        # Five pillar ABCs + canonical data models
    ├── components/        # Batteries-included implementations
    │   ├── acquirers/     # Native file readers, etc.
    │   ├── processors/    # Compressors, rerankers …
    │   ├── constructors/  # Few-shot selectors
    │   ├── evaluators/    # ACE reflector integrator
    │   └── evolvers/      # ACE curator + playbook evolver
    ├── models/            # Client abstractions + providers (API, transformers, RWKV)
    ├── methods/           # Composite closed-loop recipes (ACE closed loop, ...)
    ├── adapters/          # LangChain/LlamaIndex adapters (thin wrappers)
    ├── core/              # LLM clients + ClosedLoopOrchestrator
    └── ace/               # Original ACE reproduction (generator/reflector/curator/playbook)
```

Scripts in `scripts/` show end-to-end examples, while `tests/` cover the orchestrator, ACE wrappers, and the legacy adapters.

## Using `uv`

This repo is managed with [`uv`](https://github.com/astral-sh/uv). Typical workflow:

```bash
# Install deps
uv sync

# Run the test suite
uv run pytest

# Format/lint (optional if you add ruff/black)
uv run ruff check
```

All code lives under `src/`, so editable installs (`uv pip install -e .`) just work if you prefer a global environment.

## Minimal Closed-Loop Example

```python
from opence.core import ClosedLoopOrchestrator, DummyLLMClient
from opence.components import (
    FileSystemAcquirer,
    FewShotConstructor,
    SimpleTruncationProcessor,
    KeywordBoostReranker,
    ACEReflectorEvaluator,
    ACECuratorEvolver,
)
from opence.methods.ace import Playbook, Reflector, Curator
from opence.interfaces import LLMRequest

playbook = Playbook()
reflector_llm = DummyLLMClient()
curator_llm = DummyLLMClient()

# Queue deterministic ACE role outputs (see tests for full mocks)
# ...

orchestrator = ClosedLoopOrchestrator(
    llm=DummyLLMClient(),
    acquirer=FileSystemAcquirer("docs"),
    processors=[KeywordBoostReranker(["safety", "fire"]), SimpleTruncationProcessor()],
    constructor=FewShotConstructor(),
    evaluator=ACEReflectorEvaluator(Reflector(reflector_llm), playbook),
    evolver=ACECuratorEvolver(Curator(curator_llm), playbook),
)

result = orchestrator.run(LLMRequest(question="How to investigate industrial fires?"))
print(result.evaluation.feedback)
print(playbook.as_prompt())
```

Swap out any pillar with your own implementation (or a third-party adapter) to experiment with different CE strategies.

## Methods Layer

Many CE techniques require coordinated component bundles. The `opence.methods` package provides plug-and-play recipes, beginning with `ACEClosedLoopMethod`, which wires the ACE reflector/curator (evaluation + evolution) with any acquirer/processor/constructor you supply. Methods return fully configured `ClosedLoopOrchestrator` instances plus metadata, so higher-level runners or CLIs can let users pick `--method ace.closed_loop` and instantly inherit sensible defaults.

```python
from opence import DummyLLMClient
from opence.methods import ACEClosedLoopMethod

method = ACEClosedLoopMethod(
    generator_llm=DummyLLMClient(),
    reflector_llm=DummyLLMClient(),
    curator_llm=DummyLLMClient(),
)
orchestrator = method.build().orchestrator
```

`MethodRegistry` enables registering custom methods so downstream tooling can discover everything available in the toolkit.

## Model Providers

`opence.models` now exposes a provider layer that unifies API-based models (`OpenAIModelProvider`), local transformers (`TransformersModelProvider`), RWKV weights (`RWKVModelProvider`), and deterministic test doubles (`DummyModelProvider`). Each provider yields an `LLMClient`; the `ClosedLoopOrchestrator` automatically accepts either a raw `LLMClient` or a provider instance, keeping execution uniform regardless of backend.

## ACE Method (Legacy + Building Block)

The original ACE reproduction now lives under `opence.methods.ace`. You still get:

- `OfflineAdapter` and `OnlineAdapter` orchestration loops.
- `Playbook`, `Generator`, `Reflector`, `Curator`, and semantic deduplication utilities.
- Example scripts (`scripts/run_local_adapter.py`, `scripts/run_questions.py`) updated to import `opence.methods.ace`.

You can continue running the classic ACE training scripts:

```bash
uv run python scripts/run_local_adapter.py --model-path /path/to/model
```

The new `ACEReflectorEvaluator` + `ACECuratorEvolver` bridge these components into the generic five-pillar orchestrator, so future CE techniques can co-exist with ACE’s evolution dynamics.

## Roadmap

- **v0.1** – Deliver the closed-loop skeleton (this refactor), document interfaces, publish ACE wrappers ✅
- **v0.3** – Add more batteries (compression, dynamic few-shot, scoring adapters, `opence.contrib` registry).
- **v0.5** – Provide benchmark packs + configuration-driven pipelines; ship LangChain/LlamaIndex adapters.
- **v1.0** – Promote OpenCE to a community standard with deep OSS ecosystem integrations.

Contributions are welcome across research, engineering, evaluations, and docs. Join us in defining the future of Context Engineering! 
