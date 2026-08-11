from __future__ import annotations

import copy
import json
import os
import shlex
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any, Mapping

from .remote_training import RemoteTrainingConfig, validate_remote_training_config


_METRICS_MARKER = "__ALPHABRAIN_REMOTE_METRICS__="
_MAX_OUTPUT_BYTES = 2 * 1024 * 1024
_REMOTE_PROBE = r"""
import csv
import json
import os
import shutil
import socket
import subprocess
import time


def cpu_times():
    with open("/proc/stat", encoding="utf-8") as handle:
        values = [int(value) for value in handle.readline().split()[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def read_memory_stat(path):
    values = {}
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                key, raw = line.split(None, 1)
                values[key] = int(raw)
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return values


def cgroup_memory_snapshot(host_total):
    candidates = [
        (
            "cgroup_v2",
            "/sys/fs/cgroup/memory.current",
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory.stat",
            ("inactive_file",),
        ),
        (
            "cgroup_v1",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.stat",
            ("total_inactive_file", "inactive_file"),
        ),
    ]
    for source, current_path, limit_path, stat_path, inactive_keys in candidates:
        try:
            with open(current_path, encoding="utf-8") as handle:
                current = int(handle.read().strip())
            with open(limit_path, encoding="utf-8") as handle:
                raw_limit = handle.read().strip()
            if raw_limit == "max":
                continue
            limit = int(raw_limit)
        except (FileNotFoundError, OSError, ValueError):
            continue
        # cgroup v1 represents an unlimited hierarchy with a value close to
        # LONG_MAX. In that case /proc/meminfo is the meaningful source.
        if limit <= 0 or (host_total and limit > host_total * 2):
            continue
        stats = read_memory_stat(stat_path)
        inactive_file = next((stats[key] for key in inactive_keys if key in stats), 0)
        # Match the cAdvisor/Kubernetes working-set definition used by most
        # container dashboards: reclaimable inactive file cache is available,
        # while active cache remains part of the working set.
        used = max(0, min(limit, current - inactive_file))
        available = max(0, limit - used)
        return {
            "total_bytes": limit,
            "used_bytes": used,
            "available_bytes": available,
            "percent": round(used / limit * 100, 1),
            "source": source,
        }
    return None


def memory_snapshot():
    values = {}
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
    total = values.get("MemTotal", 0)
    cgroup = cgroup_memory_snapshot(total)
    if cgroup is not None:
        return cgroup
    available = values.get("MemAvailable", values.get("MemFree", 0))
    used = max(0, total - available)
    return {
        "total_bytes": total,
        "used_bytes": used,
        "available_bytes": available,
        "percent": round(used / total * 100, 1) if total else 0.0,
        "source": "host",
    }


def gpu_snapshot(configured_ids):
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=4,
    )
    process_counts = {}
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        timeout=4,
    )
    if processes.returncode == 0:
        for row in csv.reader(processes.stdout.splitlines()):
            if len(row) >= 2 and row[0].strip().startswith("GPU-"):
                uuid = row[0].strip()
                process_counts[uuid] = process_counts.get(uuid, 0) + 1
    result = []
    for row in csv.reader(query.stdout.splitlines()):
        if len(row) < 8:
            continue
        index = int(row[0].strip())
        if configured_ids and index not in configured_ids:
            continue
        uuid = row[1].strip()
        process_count = process_counts.get(uuid, 0)
        result.append(
            {
                "index": index,
                "uuid": uuid,
                "name": row[2].strip(),
                "memory_total_bytes": int(float(row[3].strip())) * 1024 * 1024,
                "memory_used_bytes": int(float(row[4].strip())) * 1024 * 1024,
                "memory_free_bytes": int(float(row[5].strip())) * 1024 * 1024,
                "utilization_percent": int(float(row[6].strip())),
                "temperature_c": int(float(row[7].strip())),
                "processes": [{"pid": 0} for _ in range(process_count)],
            }
        )
    return result


total_before, idle_before = cpu_times()
time.sleep(0.1)
total_after, idle_after = cpu_times()
total_delta = max(1, total_after - total_before)
idle_delta = max(0, idle_after - idle_before)
load_one, load_five, load_fifteen = os.getloadavg()
configured_ids = {
    int(value) for value in os.environ.get("ALPHABRAIN_REMOTE_GPU_IDS", "").split(",") if value.strip()
}
gpu_error = ""
try:
    gpus = gpu_snapshot(configured_ids)
except Exception as error:
    gpus = []
    gpu_error = str(error)
disk = shutil.disk_usage(os.environ["ALPHABRAIN_REMOTE_METRICS_PATH"])
payload = {
    "hostname": socket.gethostname(),
    "system_metrics": {
        "available": True,
        "cpu_percent": round(max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100)), 1),
        "load": {
            "one_minute": load_one,
            "five_minutes": load_five,
            "fifteen_minutes": load_fifteen,
        },
        "memory": memory_snapshot(),
    },
    "storage": {
        "path": os.environ["ALPHABRAIN_REMOTE_METRICS_PATH"],
        "total_bytes": disk.total,
        "used_bytes": disk.used,
        "free_bytes": disk.free,
        "low_space": False,
    },
    "gpus": gpus,
    "gpu_error": gpu_error,
}
print("__ALPHABRAIN_REMOTE_METRICS__=" + json.dumps(payload, separators=(",", ":")))
"""


def _probe_script(config: RemoteTrainingConfig) -> bytes:
    lines = [
        "set -e",
        f"cd {shlex.quote(config.repo_root)}",
        "if [ -f .env ]; then set -a; . ./.env; set +a; fi",
    ]
    if config.setup_command:
        lines.append(config.setup_command)
    lines.extend(
        [
            f"export ALPHABRAIN_REMOTE_METRICS_PATH={shlex.quote(config.repo_root)}",
            f"export ALPHABRAIN_REMOTE_GPU_IDS={shlex.quote(','.join(str(value) for value in config.gpu_ids))}",
            "python3 - <<'PY'",
            _REMOTE_PROBE.strip(),
            "PY",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _error_snapshot(config: RemoteTrainingConfig, code: str, message: str) -> dict[str, Any]:
    return {
        "enabled": True,
        "available": False,
        "target": config.target,
        "hostname": config.host,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "stale": False,
        "gpus": [],
        "system_metrics": None,
        "storage": None,
        "error": {"code": code, "message": message},
    }


class RemoteMetricsCollector:
    """Collect and briefly cache one remote training server snapshot."""

    def __init__(self, *, ttl_seconds: float = 8.0, timeout_seconds: float = 8.0) -> None:
        self.ttl_seconds = ttl_seconds
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._cached_key: tuple[Any, ...] | None = None
        self._cached_at = 0.0
        self._cached: dict[str, Any] | None = None

    @staticmethod
    def _key(config: RemoteTrainingConfig) -> tuple[Any, ...]:
        return (
            config.host,
            config.user,
            config.port,
            config.repo_root,
            config.identity_file,
            config.gpu_ids,
            config.setup_command,
        )

    def snapshot(
        self,
        config: RemoteTrainingConfig,
        reservations: Mapping[int, str] | None = None,
    ) -> dict[str, Any]:
        if not config.enabled:
            return {"enabled": False}
        validate_remote_training_config(config)
        key = self._key(config)
        now = time.monotonic()
        with self._lock:
            if self._cached_key == key and self._cached is not None and now - self._cached_at < self.ttl_seconds:
                return self._with_reservations(self._cached, reservations)
            previous = copy.deepcopy(self._cached) if self._cached_key == key and self._cached is not None else None
            try:
                command = config.ssh_command(connect_timeout=min(5, max(1, int(self.timeout_seconds))))
                completed = subprocess.run(
                    command,
                    input=_probe_script(config),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                    check=False,
                    timeout=self.timeout_seconds,
                )
                output = completed.stdout or b""
                if len(output) > _MAX_OUTPUT_BYTES:
                    raise ValueError("remote metrics output exceeded the size limit")
                text = output.decode("utf-8", errors="replace")
                marker_line = next(
                    (line for line in reversed(text.splitlines()) if line.startswith(_METRICS_MARKER)),
                    "",
                )
                if completed.returncode != 0 or not marker_line:
                    detail = (
                        text.strip().splitlines()[-1]
                        if text.strip()
                        else f"SSH exited with code {completed.returncode}"
                    )
                    raise RuntimeError(detail[:500])
                payload = json.loads(marker_line[len(_METRICS_MARKER) :])
                if not isinstance(payload, dict):
                    raise ValueError("remote metrics payload is not an object")
                result = {
                    "enabled": True,
                    "available": True,
                    "target": config.target,
                    "hostname": str(payload.get("hostname") or config.host),
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "stale": False,
                    "gpus": payload.get("gpus") if isinstance(payload.get("gpus"), list) else [],
                    "system_metrics": payload.get("system_metrics"),
                    "storage": payload.get("storage"),
                    "gpu_error": str(payload.get("gpu_error") or ""),
                    "error": None,
                }
            except subprocess.TimeoutExpired:
                result = _error_snapshot(config, "remote_metrics_timeout", "The SSH metrics probe timed out.")
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                result = _error_snapshot(config, "remote_metrics_unavailable", str(error))
            if not result["available"] and previous is not None and previous.get("available"):
                previous["stale"] = True
                previous["error"] = result["error"]
                result = previous
            self._cached_key = key
            self._cached_at = now
            self._cached = copy.deepcopy(result)
            return self._with_reservations(result, reservations)

    @staticmethod
    def _with_reservations(
        snapshot: dict[str, Any],
        reservations: Mapping[int, str] | None,
    ) -> dict[str, Any]:
        result = copy.deepcopy(snapshot)
        reserved = reservations or {}
        for gpu in result.get("gpus", []):
            if not isinstance(gpu, dict):
                continue
            index = int(gpu.get("index", -1))
            reservation = reserved.get(index)
            gpu["reserved_by_job_id"] = reservation
            processes = gpu.get("processes") if isinstance(gpu.get("processes"), list) else []
            gpu["available"] = reservation is None and not processes and not gpu.get("error")
        return result
