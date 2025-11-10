# OpenCE: The Open Context Engineering Toolkit

[English](https://github.com/sci-m-wang/OpenCE/blob/main/README.md) | [中文](https://github.com/sci-m-wang/OpenCE/blob/main/README_ZH.md)

### 🚀 Project Evolution: From `ACE-open` to `OpenCE`

Welcome\! This project is undergoing an exciting evolution.

This repository began as **`ACE-open`**, a community-driven reproduction of the **Agentic Context Engineering (ACE)** paper (arXiv:2510.04618), which had not released its official code. Thanks to the community's incredible support, it quickly gained **300+ stars**\! (Thank you\! 🙏)

The numerous Issues, discussions, and forks made one thing clear: the community doesn't just need a single reproduction. We need a robust, standardized, and extensible **Toolkit for Context Engineering** (CE).

Therefore, this project is officially evolving. We are launching **OpenCE**: a new, community-driven project to build the definitive open-source toolkit for Context Engineering, with the original ACE reproduction as its first core module.

### 🌟 The OpenCE Vision

**OpenCE (Open Context Engineering)** aims to be a modular, powerful, and easy-to-use toolkit to help developers and researchers implement, evaluate, and combine cutting-edge CE techniques.

**Our Core Principles:**

  * **Modular:** Easily swap and combine different CE strategies (RAG, Compression, Prompting).
  * **Evaluation-Driven:** Provide standardized benchmarks to measure which CE strategies *actually* work.
  * **Community-Owned:** This is not "my" project; it's "our" project.

### 🗺️ Roadmap

  * **[v0.1 - Refactor]** (In Progress)
      * [ ] Refactor the existing ACE code into the first core module: `opence.ace`.
      * [ ] Establish a clear `CONTRIBUTING.md` guide.
      * [ ] Migrate and address key Issues from the original `ACE-open` repo.
  * **[v0.5 - Core Modules]**
      * [ ] Add new modules for `opence.compression` (Context Compression).
      * [ ] Introduce `opence.evaluation` (A basic CE evaluation framework).
  * **[v1.0 - Ecosystem]**
      * [ ] Deep integration with LangChain / LlamaIndex.
      * [ ] ... and more, as decided by the community\!

### 🤝 We Need You\! (Call for Contributions)

One person can go fast, but a community can go far. To make OpenCE a reality, we need your help.

We are looking for:

  * **Developers** (to build new features and fix bugs)
  * **Researchers** (to help us integrate the latest CE papers)
  * **Doc Writers** (to make OpenCE easy to use)

**How to Start:**

1.  Check our new **[CONTRIBUTING.md](link-to-contributing-guide)** (coming soon).
2.  Look for issues tagged **[Good First Issue](link-to-issues)**.

-----

## Core Module: Agentic Context Engineering (ACE) Framework

*(This is the reproduction that started it all)*

This module is an implementation scaffold for the **Agentic Context Engineering (ACE)** method from [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models (arXiv:2510.04618)](https://arxiv.org/abs/2510.04618).

The code follows the paper’s design:

  * Contexts are structured playbooks made of bullet entries with helpful/harmful counters.
  * Three agentic roles (Generator, Reflector, Curator) interact through incremental delta updates.
  * Offline and online adaptation loops support multi-epoch training and test-time continual learning.

Refer to [docs/method\_outline.md](https://github.com/sci-m-wang/OpenCE/blob/main/docs/method_outline.md) for a distilled summary of the methodology extracted from the paper.

### Repository Layout

```
ace/         # Renamed to opence/ace in v0.1: core library modules
tests/       # Lightweight regression tests
docs/        # Engineering notes on the paper’s method
scripts/     # Example run scripts (NEW)
```

### Quick Start

Ensure Python 3.9+ (development used 3.12).

(Optional) Create a virtual environment and activate it.

Run the unit tests:

```bash
python -m unittest discover -s tests
```

### Example Usage

Here is a minimal offline adaptation loop with the dummy LLM:

```python
import json
from ace import (
    Playbook, DummyLLMClient, Generator, Reflector, Curator,
    OfflineAdapter, Sample, TaskEnvironment, EnvironmentResult
)

class ToyEnv(TaskEnvironment):
    def evaluate(self, sample, generator_output):
        gt = sample.ground_truth or ""
        pred = generator_output.final_answer
        feedback = "correct" if pred == gt else f"expected {gt} but got {pred}"
        return EnvironmentResult(feedback=feedback, ground_truth=gt)

client = DummyLLMClient()

# Queue up the expected responses for the 3 agentic roles
client.queue(json.dumps({"reasoning": "...", "bullet_ids": [], "final_answer": "42"}))
client.queue(json.dumps({"reasoning": "...", "error_identification": "", "root_cause_analysis": "",
                         "correct_approach": "", "key_insight": "Remember 42.", "bullet_tags": []}))
client.queue(json.dumps({"reasoning": "...", "operations": [{"type": "ADD", "section": "defaults",
                         "content": "Answer 42 when in doubt.", "metadata": {"helpful": 1}}]}))

adapter = OfflineAdapter(
    playbook=Playbook(),
    generator=Generator(client),
    reflector=Reflector(client),
    curator=Curator(client),
)
samples = [Sample(question="Life?", ground_truth="42")]

adapter.run(samples, ToyEnv(), epochs=1)
```

### Extending to Full Experiments

1.  **Implement an `LLMClient` subclass** that wraps your chosen model API (e.g., OpenAI, DeepSeek).
2.  **Provide task-specific prompts** (see `ace/prompts.py`) or customize them per domain.
3.  **Build `TaskEnvironment` adapters** that run the benchmark workflow (e.g., AppWorld ReAct agent, FiNER/Formula evaluation).
4.  **Configure loops:** Use `OfflineAdapter.run` and `OnlineAdapter.run` with multiple epochs as reported in the paper.
5.  **Swap in a real LLM:** For example, to use local weights on specific GPUs:
    ```bash
    CUDA_VISIBLE_DEVICES=2,3 python scripts/run_local_adapter.py
    ```
    (See `scripts/` for a minimal setup.)
