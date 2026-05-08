"""Ollama backend — uses the Ollama HTTP API with streaming."""

from __future__ import annotations

import json
from typing import Iterator

import httpx

from ..config import get_config
from .base import BackendError, BaseBackend, StreamChunk


class OllamaBackend(BaseBackend):
    name = "ollama"

    def __init__(self, host: str | None = None) -> None:
        cfg = get_config()
        self._base = (host or cfg.ollama_host).rstrip("/")
        self._timeout = cfg.ollama_timeout

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self._base}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            r = httpx.get(f"{self._base}/api/tags", timeout=10)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception as exc:
            raise BackendError(f"Ollama list_models failed: {exc}") from exc

    # ------------------------------------------------------------------
    def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> tuple[str, int, int]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        try:
            r = httpx.post(
                f"{self._base}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            r.raise_for_status()
            data = r.json()
            return (
                data.get("response", ""),
                data.get("prompt_eval_count", 0),
                data.get("eval_count", 0),
            )
        except httpx.HTTPStatusError as exc:
            raise BackendError(f"Ollama HTTP error: {exc}") from exc
        except Exception as exc:
            raise BackendError(f"Ollama generate failed: {exc}") from exc

    # ------------------------------------------------------------------
    def stream(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Iterator[StreamChunk]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        try:
            with httpx.stream(
                "POST",
                f"{self._base}/api/generate",
                json=payload,
                timeout=self._timeout,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    data = json.loads(raw_line)
                    done = data.get("done", False)
                    yield StreamChunk(
                        text=data.get("response", ""),
                        is_final=done,
                        prompt_tokens=data.get("prompt_eval_count", 0) if done else 0,
                        output_tokens=data.get("eval_count", 0) if done else 0,
                    )
                    if done:
                        break
        except httpx.HTTPStatusError as exc:
            raise BackendError(f"Ollama stream HTTP error: {exc}") from exc
        except Exception as exc:
            raise BackendError(f"Ollama stream failed: {exc}") from exc
