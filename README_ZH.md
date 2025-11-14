# OpenCE：闭环上下文工程工具箱

OpenCE 是一个可插拔的**闭环上下文工程 (Closed-Loop CE)** 元框架，将社区版 ACE 复现升级为能够“感知 → 构建 → 评估 → 进化”的自我进化系统。

## 为什么选择闭环？

传统 RAG 是开环流程——获取一次上下文就立即回答。OpenCE 在末端新增两步：

1. **运行时评估**：每次 LLM 回复都会被自动评估（例如 ACE Reflector、RAGAS）。
2. **策略进化**：评估信号会驱动策略或记忆库的更新（例如 ACE Curator 更新 Playbook）。

这样形成一个不断自我强化的“闭环飞轮”。

## 五大支柱架构

OpenCE 将闭环拆分为五个接口（位于 `src/opence/interfaces/`）：

| 支柱 | 接口 | 职责 |
| --- | --- | --- |
| 获取 (Acquisition) | `IAcquirer` | 感知层，负责从 DB/文件/Web/LangChain 拉取原始信息。 |
| 处理 (Processing) | `IProcessor` | 对信息进行清洗、切分、压缩或重排序。 |
| 构建 (Construction) | `IConstructor` | 将处理后的信息组装成 Prompt/Few-shot 上下文。 |
| 评估 (Evaluation) | `IEvaluator` | 对 LLM 响应进行质检，产生反馈信号。 |
| 进化 (Evolution) | `IEvolver` | 消耗评估信号，更新长期策略/记忆库。 |

接口是“灵魂 (Soul)”，原生组件位于 `src/opence/components/`（“电池 (Batteries)”），第三方适配器在 `src/opence/adapters/`（“胶水 (Glue)”），而 `src/opence/core/orchestrator.py` 则是驱动整个闭环的“引擎”。

## 代码结构

```
src/
└── opence/
    ├── interfaces/        # 抽象接口 + Pydantic 数据模型
    ├── components/        # 原生组件：acquirers/processors/constructors/evaluators/evolvers
    ├── models/            # 模型客户端与 Provider（API、本地 transformers、RWKV）
    ├── methods/           # 综合方法（如 ACE 闭环）
    ├── adapters/          # LangChain 等生态的薄封装
    ├── core/              # LLM Client + ClosedLoopOrchestrator
    └── ace/               # 原 ACE 复现，现作为 Evolver/Evaluator 子模块
```

`scripts/` 提供端到端示例，`tests/` 覆盖 orchestrator 与 ACE 封装。

## 使用 `uv`

项目使用 [`uv`](https://github.com/astral-sh/uv) 管理依赖：

```bash
uv sync             # 安装依赖
uv run pytest       # 运行测试
uv run python scripts/run_local_adapter.py  # 运行示例脚本
```

所有源码位于 `src/`，如需全局安装可直接运行 `uv pip install -e .`。

## 闭环示例

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
reflector = Reflector(DummyLLMClient())
curator = Curator(DummyLLMClient())

orchestrator = ClosedLoopOrchestrator(
    llm=DummyLLMClient(),
    acquirer=FileSystemAcquirer("docs"),
    processors=[KeywordBoostReranker(["安全", "火灾"]), SimpleTruncationProcessor()],
    constructor=FewShotConstructor(),
    evaluator=ACEReflectorEvaluator(reflector, playbook),
    evolver=ACECuratorEvolver(curator, playbook),
)

result = orchestrator.run(LLMRequest(question="如何开展工业火灾勘验？"))
print(result.evaluation.feedback)
print(playbook.as_prompt())
```

任一支柱都可以用你自己的实现或第三方适配器替换，构建不同的 CE 策略组合。

## 方法层（Methods）

`opence.methods` 提供“综合方法”定义，用于一次性装配多个组件。首个实现 `ACEClosedLoopMethod` 将 ACE 的 Reflector（评估）与 Curator（进化）封装为可直接调用的闭环方案：

```python
from opence import DummyLLMClient
from opence.methods import ACEClosedLoopMethod

method = ACEClosedLoopMethod(
    generator_llm=DummyLLMClient(),
    reflector_llm=DummyLLMClient(),
    curator_llm=DummyLLMClient(),
)
loop = method.build().orchestrator
```

通过 `MethodRegistry` 可以注册/发现自定义方法，让 CLI 或服务层按名称启用。

## 模型层（Models）

`opence.models` 增加了 Provider 抽象，统一 API 模型（`OpenAIModelProvider`）、本地 transformers（`TransformersModelProvider`）、RWKV 权重（`RWKVModelProvider`）以及测试用的 `DummyModelProvider`。`ClosedLoopOrchestrator` 支持直接接收 Provider 或已有的 `LLMClient`，从而在不同模型后端之间保持一致的调用体验。

## ACE 模块

原始 ACE 复现现位于 `opence.methods.ace`，依旧提供：

- `OfflineAdapter` / `OnlineAdapter`
- `Playbook`、`Generator`、`Reflector`、`Curator`、语义去重
- 更新后的脚本 `scripts/run_local_adapter.py`、`scripts/run_questions.py`

示例运行方式：

```bash
uv run python scripts/run_local_adapter.py --model-path /path/to/model
```

`ACEReflectorEvaluator` + `ACECuratorEvolver` 已经把这些角色桥接到新的闭环 orchestrator 中，让 ACE 成为 toolkit 的首个 Evolver/Evaluator 实现。

## 路线图

- **v0.1**：完成闭环骨架（当前版本），发布 ACE 适配组件。
- **v0.3**：引入更多电池（压缩、动态 few-shot、打分适配器、`opence.contrib` 注册表）。
- **v0.5**：提供配置化的 pipeline + 基准套件，强化 LangChain/LlamaIndex 适配。
- **v1.0**：形成社区标准，深入对接更广泛的开放生态。

欢迎开发者、研究者和文档贡献者加入，共同打造下一代上下文工程体系！
