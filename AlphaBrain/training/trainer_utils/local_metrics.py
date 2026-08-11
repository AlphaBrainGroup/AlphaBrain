"""Best-effort append-only local training metrics.

The UI consumes ``metrics.jsonl`` while a job is running.  This helper is
intentionally stdlib-only so trainers can emit the same small event schema
without coupling their logging path to W&B or to a heavyweight ML package.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from numbers import Number
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RESERVED_FIELDS = frozenset({"timestamp", "phase", "step", "iteration"})


def _as_json_number(value: Any) -> int | float | None:
    """Return a finite JSON number for Python, NumPy, or tensor scalars."""
    if isinstance(value, bool):
        return int(value)

    if not isinstance(value, Number):
        item = getattr(value, "item", None)
        if not callable(item):
            return None
        try:
            value = item()
        except (TypeError, ValueError, RuntimeError):
            return None

    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, Number) or isinstance(value, complex):
        return None

    if isinstance(value, int):
        return value

    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _numeric_metrics(metrics: Mapping[Any, Any], prefix: str = "") -> dict[str, int | float]:
    """Flatten nested mappings and retain only finite scalar metrics."""
    result: dict[str, int | float] = {}
    for raw_key, value in metrics.items():
        key = str(raw_key)
        full_key = f"{prefix}/{key}" if prefix else key
        if not prefix and full_key in _RESERVED_FIELDS:
            continue
        if isinstance(value, Mapping):
            result.update(_numeric_metrics(value, full_key))
            continue
        numeric = _as_json_number(value)
        if numeric is not None:
            result[full_key] = numeric
    return result


def append_local_metrics(
    output_dir: str | os.PathLike[str],
    metrics: Mapping[Any, Any],
    *,
    phase: str,
    step: int | None = None,
    iteration: int | None = None,
) -> bool:
    """Append one event to ``<output_dir>/metrics.jsonl``.

    Writes are best-effort: an instrumentation failure is logged and returns
    ``False`` rather than interrupting training.  Calls should be limited to
    the main process for distributed trainers.
    """
    if not isinstance(phase, str) or not phase.strip():
        logger.warning("Skipping local metrics event with an empty phase")
        return False
    if step is None and iteration is None:
        logger.warning("Skipping local metrics event without a step or iteration")
        return False

    event: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "phase": phase,
    }
    if step is not None:
        numeric_step = _as_json_number(step)
        if numeric_step is None:
            logger.warning("Skipping local metrics event with invalid step %r", step)
            return False
        event["step"] = int(numeric_step)
    if iteration is not None:
        numeric_iteration = _as_json_number(iteration)
        if numeric_iteration is None:
            logger.warning("Skipping local metrics event with invalid iteration %r", iteration)
            return False
        event["iteration"] = int(numeric_iteration)
    event["metrics"] = _numeric_metrics(metrics)

    metrics_path = Path(output_dir) / "metrics.jsonl"
    try:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        fd = os.open(metrics_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Could not append local metrics to %s: %s", metrics_path, exc)
        return False
    return True
