"""Best-effort host CPU and memory telemetry for the Web UI dashboard."""

from __future__ import annotations

from typing import Any

import psutil


def collect_system_metrics() -> dict[str, Any]:
    """Collect a non-blocking host snapshot without breaking dashboard requests.

    ``psutil.cpu_percent(interval=None)`` reports the percentage since the
    previous call and, unlike a positive interval, never pauses the request.
    Consumers must check ``available`` because host telemetry can be denied by
    a container or operating-system policy.
    """

    try:
        memory = psutil.virtual_memory()
        load_one, load_five, load_fifteen = psutil.getloadavg()
        return {
            "available": True,
            "cpu_percent": float(psutil.cpu_percent(interval=None)),
            "load": {
                "one_minute": float(load_one),
                "five_minutes": float(load_five),
                "fifteen_minutes": float(load_fifteen),
            },
            "memory": {
                "total_bytes": int(memory.total),
                "used_bytes": int(memory.used),
                "available_bytes": int(memory.available),
                "percent": float(memory.percent),
            },
        }
    except Exception as exc:  # Host telemetry is optional and must never break the dashboard.
        return {
            "available": False,
            "cpu_percent": None,
            "load": None,
            "memory": None,
            "error": {
                "code": "system_metrics_unavailable",
                "message": str(exc),
            },
        }
