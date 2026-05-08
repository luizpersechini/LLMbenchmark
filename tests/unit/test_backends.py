"""Unit tests for backends using mocked HTTP responses."""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from llm_benchmark.backends.base import BackendError, StreamChunk
from llm_benchmark.backends.ollama import OllamaBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stream_response(tokens: list[str], model: str = "test") -> list[bytes]:
    """Build NDJSON bytes simulating an Ollama streaming response."""
    lines = []
    for i, tok in enumerate(tokens):
        is_last = i == len(tokens) - 1
        obj = {"response": tok, "done": is_last}
        if is_last:
            obj["prompt_eval_count"] = 5
            obj["eval_count"] = len(tokens)
        lines.append(json.dumps(obj).encode())
    return lines


# ---------------------------------------------------------------------------
# OllamaBackend.is_available
# ---------------------------------------------------------------------------

class TestOllamaIsAvailable:
    def test_returns_true_when_200(self):
        with patch("llm_benchmark.backends.ollama.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert OllamaBackend().is_available() is True

    def test_returns_false_on_connection_error(self):
        with patch("llm_benchmark.backends.ollama.httpx.get", side_effect=Exception("refused")):
            assert OllamaBackend().is_available() is False

    def test_returns_false_when_non_200(self):
        with patch("llm_benchmark.backends.ollama.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=500)
            assert OllamaBackend().is_available() is False


# ---------------------------------------------------------------------------
# OllamaBackend.list_models
# ---------------------------------------------------------------------------

class TestOllamaListModels:
    def test_parses_model_names(self):
        payload = {"models": [{"name": "llama3:8b"}, {"name": "gemma:2b"}]}
        with patch("llm_benchmark.backends.ollama.httpx.get") as mock_get:
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = payload
            resp.raise_for_status = MagicMock()
            mock_get.return_value = resp
            models = OllamaBackend().list_models()
        assert models == ["llama3:8b", "gemma:2b"]

    def test_raises_backend_error_on_exception(self):
        with patch("llm_benchmark.backends.ollama.httpx.get", side_effect=Exception("network")):
            with pytest.raises(BackendError):
                OllamaBackend().list_models()

    def test_returns_empty_when_no_models(self):
        with patch("llm_benchmark.backends.ollama.httpx.get") as mock_get:
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"models": []}
            resp.raise_for_status = MagicMock()
            mock_get.return_value = resp
            assert OllamaBackend().list_models() == []


# ---------------------------------------------------------------------------
# OllamaBackend.generate
# ---------------------------------------------------------------------------

class TestOllamaGenerate:
    def test_returns_output_and_token_counts(self):
        payload = {
            "response": "Hello world",
            "prompt_eval_count": 3,
            "eval_count": 2,
        }
        with patch("llm_benchmark.backends.ollama.httpx.post") as mock_post:
            resp = MagicMock()
            resp.json.return_value = payload
            resp.raise_for_status = MagicMock()
            mock_post.return_value = resp
            output, pt, ot = OllamaBackend().generate("test", "hi")
        assert output == "Hello world"
        assert pt == 3
        assert ot == 2

    def test_raises_backend_error_on_failure(self):
        with patch("llm_benchmark.backends.ollama.httpx.post", side_effect=Exception("net")):
            with pytest.raises(BackendError):
                OllamaBackend().generate("test", "hi")

    def test_passes_correct_options(self):
        with patch("llm_benchmark.backends.ollama.httpx.post") as mock_post:
            resp = MagicMock()
            resp.json.return_value = {"response": "", "prompt_eval_count": 0, "eval_count": 0}
            resp.raise_for_status = MagicMock()
            mock_post.return_value = resp
            OllamaBackend().generate("mymodel", "prompt", max_tokens=512, temperature=0.7)
            call_kwargs = mock_post.call_args
            body = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
            assert body["options"]["num_predict"] == 512
            assert body["options"]["temperature"] == pytest.approx(0.7)
            assert body["model"] == "mymodel"


# ---------------------------------------------------------------------------
# OllamaBackend.stream
# ---------------------------------------------------------------------------

class TestOllamaStream:
    def _mock_stream_ctx(self, lines: list[bytes]):
        """Build a context manager mock that iterates lines."""
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        ctx.raise_for_status = MagicMock()
        ctx.iter_lines = MagicMock(return_value=(l.decode() for l in lines))
        return ctx

    def test_yields_chunks_and_final(self):
        lines = _make_stream_response(["He", "llo", " world"])
        with patch("llm_benchmark.backends.ollama.httpx.stream") as mock_stream:
            mock_stream.return_value = self._mock_stream_ctx(lines)
            chunks = list(OllamaBackend().stream("test", "hi", max_tokens=10))
        texts = [c.text for c in chunks if c.text]
        assert "".join(texts) == "Hello world"
        assert chunks[-1].is_final is True
        assert chunks[-1].output_tokens == 3

    def test_raises_backend_error_on_exception(self):
        with patch("llm_benchmark.backends.ollama.httpx.stream", side_effect=Exception("net")):
            with pytest.raises(BackendError):
                list(OllamaBackend().stream("test", "hi"))
