import os
import threading
import time

from fastapi.testclient import TestClient

# Disable preflight so tests can import the server without mounted model files.
os.environ["MOSS_PREFLIGHT"] = "0"

import server  # noqa: E402


class FakeDaemon:
    def __init__(self):
        self._alive = True
        self.pid = 12345
        self.restart_count = 0
        self.last_error = None

    def is_alive(self) -> bool:
        return self._alive


def _configure_test_state(monkeypatch, concurrency_limit: int = 1) -> FakeDaemon:
    fake_daemon = FakeDaemon()

    def refresh() -> None:
        server._update_daemon_status(
            alive=fake_daemon.is_alive(),
            pid=fake_daemon.pid,
            last_error=fake_daemon.last_error,
            restart_count=fake_daemon.restart_count,
        )

    monkeypatch.setattr(server, "READY", True)
    monkeypatch.setattr(server, "_daemon", fake_daemon)
    monkeypatch.setattr(server, "_get_daemon", lambda: fake_daemon)
    monkeypatch.setattr(server._daemon_monitor, "start_once", lambda: None)
    monkeypatch.setattr(server._daemon_monitor, "refresh", refresh)
    monkeypatch.setattr(
        server,
        "REQUEST_SEMAPHORE",
        threading.BoundedSemaphore(concurrency_limit),
    )
    refresh()
    return fake_daemon


def test_health_stays_fast_during_generation(monkeypatch):
    _configure_test_state(monkeypatch, concurrency_limit=1)
    started = threading.Event()
    release = threading.Event()

    def slow_generate(text, reference_audio, language):
        started.set()
        assert release.wait(timeout=5), "timed out waiting to release mock generation"
        return b"fake wav bytes"

    monkeypatch.setattr(server, "generate_speech", slow_generate)

    with TestClient(server.app) as client:
        generation_result = {}

        def run_generate():
            generation_result["response"] = client.post(
                "/api/generate",
                data={"text": "hello"},
            )

        worker = threading.Thread(target=run_generate, daemon=True)
        worker.start()
        assert started.wait(timeout=2), "generate_speech mock was not called"

        began = time.perf_counter()
        response = client.get("/health")
        elapsed = time.perf_counter() - began

        assert elapsed < 0.5
        assert response.status_code == 200
        payload = response.json()
        assert payload["daemon_alive"] is True

        release.set()
        worker.join(timeout=5)
        assert generation_result["response"].status_code == 200


def test_generate_returns_429_when_above_concurrency_limit(monkeypatch):
    _configure_test_state(monkeypatch, concurrency_limit=1)
    started = threading.Event()
    release = threading.Event()

    def slow_generate(text, reference_audio, language):
        started.set()
        assert release.wait(timeout=5), "timed out waiting to release mock generation"
        return b"fake wav bytes"

    monkeypatch.setattr(server, "generate_speech", slow_generate)

    with TestClient(server.app) as client:
        first_result = {}

        def run_first_request():
            first_result["response"] = client.post(
                "/api/generate",
                data={"text": "first"},
            )

        worker = threading.Thread(target=run_first_request, daemon=True)
        worker.start()
        assert started.wait(timeout=2), "first request did not reach generate_speech"

        second = client.post("/api/generate", data={"text": "second"})
        assert second.status_code == 429

        release.set()
        worker.join(timeout=5)
        assert first_result["response"].status_code == 200
