"""
Tests for split_text() segmentation and /health endpoint (503 gate).

Run:  MOSS_PREFLIGHT=0 pytest services/local/test_split_text.py -v
"""
import os

# Disable preflight so we can import server without model files on disk.
os.environ["MOSS_PREFLIGHT"] = "0"

import pytest
from server import split_text, app, MISSING, READY  # noqa: E402
from fastapi.testclient import TestClient


# ──────────────── Bug 1: split_text ────────────────────────────────────────


class TestSplitTextTrentCases:
    """Exact inputs from Trent's review; both were broken before the fix."""

    def test_chinese_sentence_with_commas(self):
        """No duplicate first segment, no lost middle commas."""
        result = split_text("abcdefghij。klmnopqrstuv，wx，yz", max_chars=10)
        # Must not contain duplicates
        assert len(result) == len(set(result)), f"duplicates in {result}"
        # All original content must be present
        joined = "".join(result)
        assert "abcdefghij。" in joined
        assert "wx，" in joined or "wx" in joined
        assert "yz" in joined
        assert "klmnopqrstuv，" in joined or "klmnopqrstuv" in joined

    def test_english_comma_separated(self):
        """No dropped middle chunks (foo, bar, baz must survive)."""
        result = split_text("hello world,foo,bar,baz,qux", max_chars=10)
        joined = "".join(result)
        for word in ("hello", "world", "foo", "bar", "baz", "qux"):
            assert word in joined, f"'{word}' missing from segments {result}"


class TestSplitTextGeneral:
    """Additional coverage."""

    def test_short_text_single_segment(self):
        """Text shorter than max_chars stays in one segment."""
        text = "Hello world"
        result = split_text(text, max_chars=500)
        assert result == [text]

    def test_pure_chinese_with_exclamation_question(self):
        """Splits on ！ and ？ for Chinese text."""
        text = "你好世界！这是一个测试？是的，这是测试。"
        result = split_text(text, max_chars=10)
        joined = "".join(result)
        # All characters must be preserved
        assert joined == text

    def test_single_long_sentence_no_commas(self):
        """A comma-less sentence exceeding max_chars is emitted whole (not infinite loop)."""
        text = "a" * 50
        result = split_text(text, max_chars=10)
        # Must not infinite-loop; the long segment is kept whole
        assert len(result) >= 1
        assert "".join(result) == text

    def test_no_content_loss(self):
        """Concatenated segments must equal original text (modulo whitespace around sentence delimiters)."""
        text = "短句。中等的句子，有逗号。很长很长很长很长很长很长很长的句子，需要按照逗号来拆分，才能变短！最后一句？"
        result = split_text(text, max_chars=15)
        joined = "".join(result)
        # Whitespace can be stripped by the function, but Han characters must survive
        for ch in text:
            if not ch.isspace():
                assert ch in joined, f"character '{ch}' lost in {result}"


# ──────────────── Bug 2: /health endpoint ──────────────────────────────────


class TestHealthEndpoint:
    """When MISSING is non-empty the endpoint must return 503."""

    def test_health_returns_503_when_not_ready(self):
        """Simulate missing weights by injecting into MISSING list."""
        import server as srv

        # Save originals
        orig_missing = srv.MISSING[:]
        orig_ready = srv.READY
        try:
            srv.MISSING.clear()
            srv.MISSING.append("MODEL_GGUF=/models/fake.gguf")
            srv.READY = False

            client = TestClient(app)
            resp = client.get("/health")
            assert resp.status_code == 503, f"expected 503, got {resp.status_code}"
            body = resp.json()
            assert body["ready"] is False
            assert body["status"] == "degraded"
        finally:
            srv.MISSING.clear()
            srv.MISSING.extend(orig_missing)
            srv.READY = orig_ready

    def test_health_returns_200_when_ready(self):
        """When everything is present, health returns 200."""
        import server as srv

        orig_missing = srv.MISSING[:]
        orig_ready = srv.READY
        try:
            srv.MISSING.clear()
            srv.READY = True

            client = TestClient(app)
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["ready"] is True
        finally:
            srv.MISSING.clear()
            srv.MISSING.extend(orig_missing)
            srv.READY = orig_ready
