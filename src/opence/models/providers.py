"""Model provider abstractions that expose a unified LLM interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence

from .clients import (
    DummyLLMClient,
    LLMClient,
    TransformersLLMClient,
    DeepseekLLMClient,
)


class BaseModelProvider(ABC):
    """Produces configured `LLMClient` instances on demand."""

    def __init__(self) -> None:
        self._cached_client: Optional[LLMClient] = None

    def client(self) -> LLMClient:
        if self._cached_client is None:
            self._cached_client = self.create_client()
        return self._cached_client

    @abstractmethod
    def create_client(self) -> LLMClient:
        """Instantiate the underlying client."""

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Utility so providers can be used directly."""
        return self.client().complete(prompt, **kwargs).text


class OpenAIModelProvider(BaseModelProvider):
    """Wraps OpenAI-compatible chat completions endpoints."""

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = system_prompt

    def create_client(self) -> LLMClient:
        return DeepseekLLMClient(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            system_prompt=self.system_prompt,
        )


class TransformersModelProvider(BaseModelProvider):
    """Backs models hosted locally via Hugging Face `transformers`."""

    def __init__(
        self,
        *,
        model_path: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        device_map: str | Dict[str, int] = "auto",
        torch_dtype: str | "torch.dtype" = "auto",
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device_map = device_map
        self.torch_dtype = torch_dtype

    def create_client(self) -> LLMClient:
        return TransformersLLMClient(
            self.model_path,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
        )


class RWKVModelProvider(BaseModelProvider):
    """Provides access to RWKV models via the official pipeline."""

    def __init__(
        self,
        *,
        model_path: str,
        tokenizer_path: str,
        strategy: str = "cuda fp16i8 *20 -> cpu fp32",
        default_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.5,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.strategy = strategy
        self.default_max_new_tokens = default_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def create_client(self) -> LLMClient:
        from .rwkv_client import RWKVLLMClient

        return RWKVLLMClient(
            model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            strategy=self.strategy,
            max_new_tokens=self.default_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )


class DummyModelProvider(BaseModelProvider):
    """Returns the deterministic dummy client, useful for tests."""

    def __init__(self, responses: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self._responses = list(responses or [])

    def create_client(self) -> LLMClient:
        client = DummyLLMClient()
        for text in self._responses:
            client.queue(text)
        return client
