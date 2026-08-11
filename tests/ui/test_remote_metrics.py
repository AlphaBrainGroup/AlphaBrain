from __future__ import annotations

import json
import subprocess

from alphabrain_ui.remote_metrics import RemoteMetricsCollector, _REMOTE_PROBE
from alphabrain_ui.remote_training import RemoteTrainingConfig


def remote_config() -> RemoteTrainingConfig:
    return RemoteTrainingConfig(
        enabled=True,
        host="gpu.example.edu",
        user="researcher",
        repo_root="/srv/AlphaBrain",
        gpu_ids=(0, 2),
    )


def probe_output() -> bytes:
    payload = {
        "hostname": "gpu-node-01",
        "system_metrics": {
            "available": True,
            "cpu_percent": 25.0,
            "load": {"one_minute": 1.0, "five_minutes": 0.5, "fifteen_minutes": 0.25},
            "memory": {
                "total_bytes": 1000,
                "used_bytes": 400,
                "available_bytes": 600,
                "percent": 40.0,
            },
        },
        "storage": {
            "path": "/srv/AlphaBrain",
            "total_bytes": 2000,
            "used_bytes": 500,
            "free_bytes": 1500,
            "low_space": False,
        },
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-0",
                "name": "GPU zero",
                "memory_total_bytes": 100,
                "memory_used_bytes": 20,
                "memory_free_bytes": 80,
                "utilization_percent": 10,
                "temperature_c": 40,
                "processes": [],
            }
        ],
        "gpu_error": "",
    }
    return ("noise\n__ALPHABRAIN_REMOTE_METRICS__=" + json.dumps(payload) + "\n").encode()


def test_remote_metrics_are_cached_and_reservations_are_applied(monkeypatch) -> None:
    calls = []

    def completed(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=probe_output())

    monkeypatch.setattr("alphabrain_ui.remote_metrics.subprocess.run", completed)
    collector = RemoteMetricsCollector(ttl_seconds=60)

    first = collector.snapshot(remote_config(), {0: "job-1"})
    second = collector.snapshot(remote_config(), {})

    assert len(calls) == 1
    assert calls[0][0][-4:] == ["researcher@gpu.example.edu", "bash", "-s", "--"]
    assert first["hostname"] == "gpu-node-01"
    assert first["gpus"][0]["reserved_by_job_id"] == "job-1"
    assert first["gpus"][0]["available"] is False
    assert second["gpus"][0]["reserved_by_job_id"] is None
    assert second["gpus"][0]["available"] is True


def test_remote_metrics_return_structured_connection_error(monkeypatch) -> None:
    def failed(command, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(command, 255, stdout=b"ssh: connect to host failed\n")

    monkeypatch.setattr("alphabrain_ui.remote_metrics.subprocess.run", failed)
    result = RemoteMetricsCollector(ttl_seconds=0).snapshot(remote_config())

    assert result["available"] is False
    assert result["gpus"] == []
    assert result["error"] == {
        "code": "remote_metrics_unavailable",
        "message": "ssh: connect to host failed",
    }


def test_remote_probe_prefers_cgroup_working_set_memory() -> None:
    assert '"/sys/fs/cgroup/memory.current"' in _REMOTE_PROBE
    assert '"/sys/fs/cgroup/memory.max"' in _REMOTE_PROBE
    assert "current - inactive_file" in _REMOTE_PROBE
    assert '"source": source' in _REMOTE_PROBE
