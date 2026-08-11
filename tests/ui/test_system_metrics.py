from types import SimpleNamespace

from alphabrain_ui.system_metrics import collect_system_metrics


def test_collect_system_metrics_returns_cpu_load_and_memory(monkeypatch) -> None:
    monkeypatch.setattr("alphabrain_ui.system_metrics.psutil.cpu_percent", lambda *, interval: 37.5)
    monkeypatch.setattr("alphabrain_ui.system_metrics.psutil.getloadavg", lambda: (1.25, 0.75, 0.5))
    monkeypatch.setattr(
        "alphabrain_ui.system_metrics.psutil.virtual_memory",
        lambda: SimpleNamespace(total=32_000, used=12_000, available=20_000, percent=37.5),
    )

    assert collect_system_metrics() == {
        "available": True,
        "cpu_percent": 37.5,
        "load": {"one_minute": 1.25, "five_minutes": 0.75, "fifteen_minutes": 0.5},
        "memory": {
            "total_bytes": 32_000,
            "used_bytes": 12_000,
            "available_bytes": 20_000,
            "percent": 37.5,
        },
    }


def test_collect_system_metrics_returns_structured_error(monkeypatch) -> None:
    def unavailable():
        raise PermissionError("host metrics denied")

    monkeypatch.setattr("alphabrain_ui.system_metrics.psutil.virtual_memory", unavailable)

    result = collect_system_metrics()

    assert result["available"] is False
    assert result["cpu_percent"] is None
    assert result["memory"] is None
    assert result["error"] == {
        "code": "system_metrics_unavailable",
        "message": "host metrics denied",
    }
