"""MLX backend — Apple Silicon native via mlx-lm (optional dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .base import BackendError, BaseBackend, StreamChunk


class MLXBackend(BaseBackend):
    """Wraps mlx_lm.load / mlx_lm.generate for Apple Silicon inference.

    mlx-lm must be installed: pip install 'llm-benchmark[mlx]'
    Models are specified as HuggingFace repo IDs or local paths.
    """

    name = "mlx"

    def __init__(self) -> None:
        try:
            import mlx_lm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is not installed. Run: pip install 'llm-benchmark[mlx]'"
            ) from exc
        self._cache: dict[str, tuple] = {}  # model_id -> (model, tokenizer)

    def _load(self, model: str) -> tuple:
        if model not in self._cache:
            import mlx_lm
            self._cache[model] = mlx_lm.load(model)
        return self._cache[model]

    def is_available(self) -> bool:
        try:
            import mlx_lm  # noqa: F401
            return True
        except ImportError:
            return False

    def list_models(self) -> list[str]:
        """Return already-cached models plus any found in ~/.cache/huggingface."""
        models = list(self._cache.keys())
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        if hf_cache.exists():
            for p in hf_cache.iterdir():
                if p.is_dir() and p.name.startswith("models--"):
                    slug = p.name.replace("models--", "").replace("--", "/", 1)
                    if slug not in models:
                        models.append(slug)
        return sorted(models)

    def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> tuple[str, int, int]:
        try:
            import mlx_lm
            m, tok = self._load(model)
            output = mlx_lm.generate(
                m,
                tok,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False,
            )
            prompt_tokens = len(tok.encode(prompt))
            output_tokens = len(tok.encode(output))
            return output, prompt_tokens, output_tokens
        except Exception as exc:
            raise BackendError(f"MLX generate failed: {exc}") from exc

    def stream(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Iterator[StreamChunk]:
        """MLX-LM doesn't have a native streaming API; we fake it token-by-token."""
        try:
            import mlx_lm
            m, tok = self._load(model)
            prompt_ids = tok.encode(prompt)
            prompt_tokens = len(prompt_ids)
            full_output = ""
            for token_text in mlx_lm.stream_generate(
                m,
                tok,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
            ):
                full_output += token_text
                yield StreamChunk(text=token_text, is_final=False)
            output_tokens = len(tok.encode(full_output))
            yield StreamChunk(
                text="",
                is_final=True,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
        except Exception as exc:
            raise BackendError(f"MLX stream failed: {exc}") from exc
