from .base import BackendError, BaseBackend
from .ollama import OllamaBackend

__all__ = ["BaseBackend", "BackendError", "OllamaBackend"]

try:
    from .mlx import MLXBackend  # noqa: F401

    __all__.append("MLXBackend")
except ImportError:
    pass


def get_backend(name: str) -> BaseBackend:
    name = name.lower()
    if name == "ollama":
        return OllamaBackend()
    if name == "mlx":
        try:
            from .mlx import MLXBackend

            return MLXBackend()
        except ImportError as e:
            raise ImportError(
                "mlx-lm is not installed. Run: pip install 'llm-benchmark[mlx]'"
            ) from e
    raise ValueError(f"Unknown backend: {name!r}. Choose 'ollama' or 'mlx'.")
