"""LLMClient adapter for RWKV models."""

from __future__ import annotations

from typing import Any

from .clients import LLMClient, LLMResponse


class RWKVLLMClient(LLMClient):
    """Thin wrapper around the official RWKV pipeline."""

    def __init__(
        self,
        *,
        model_path: str,
        tokenizer_path: str,
        strategy: str = "cuda fp16i8 *20 -> cpu fp32",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.5,
    ) -> None:
        super().__init__(model=model_path)
        try:
            from rwkv.model import RWKV  # type: ignore
            from rwkv.utils import PIPELINE  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'rwkv' package is required for RWKVModelProvider. "
                "Install it via `pip install rwkv` and provide valid weights."
            ) from exc

        self._pipeline = PIPELINE(
            RWKV(model=model_path, strategy=strategy), tokenizer_path
        )
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        token_count = int(kwargs.get("max_new_tokens", self._max_new_tokens))
        temperature = float(kwargs.get("temperature", self._temperature))
        top_p = float(kwargs.get("top_p", self._top_p))

        text = self._pipeline.generate(
            prompt,
            token_count=token_count,
            temperature=temperature,
            top_p=top_p,
        )
        return LLMResponse(text=str(text).strip())
