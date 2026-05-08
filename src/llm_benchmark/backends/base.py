"""Abstract base class all backends must implement."""

from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass


class BackendError(Exception):
    """Raised when a backend call fails."""


@dataclass
class StreamChunk:
    text: str
    is_final: bool = False
    prompt_tokens: int = 0
    output_tokens: int = 0


class BaseBackend(abc.ABC):
    """Abstract LLM backend.

    Subclasses implement ``generate`` (blocking) and ``stream`` (iterator).
    Both must return consistent token counts so callers can time TPS correctly.
    """

    name: str = "base"

    @abc.abstractmethod
    def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> tuple[str, int, int]:
        """Return (output_text, prompt_tokens, output_tokens)."""

    @abc.abstractmethod
    def stream(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Iterator[StreamChunk]:
        """Yield StreamChunk objects; the final chunk carries token counts."""

    @abc.abstractmethod
    def list_models(self) -> list[str]:
        """Return model names available on this backend."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend service is reachable."""
