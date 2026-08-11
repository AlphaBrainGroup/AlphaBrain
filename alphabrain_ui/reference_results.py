"""Read-only, versioned reference results captured from the public website.

The runtime never fetches the website. Packaged snapshots are validated before
being exposed, and references are returned only when their benchmark and
signature constraints match the caller's local evaluation.
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import yaml

REFERENCE_SCHEMA = "alphabrain.reference-results"
REFERENCE_SCHEMA_VERSION = 1
REFERENCE_SNAPSHOT_PATH = Path(__file__).with_name("registry") / "reference_results_2026-07-16.yaml"
_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{1,159}$")
_UNITS = {"percent", "count", "steps", "score", "seconds"}
_DIRECTIONS = {"higher", "lower", "neutral"}


class ReferenceResultsError(ValueError):
    def __init__(self, code: str, path: str = "", detail: Any = None):
        self.code = code
        self.path = path
        self.detail = detail
        super().__init__(code)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReferenceResultsError("reference_mapping_required", path)
    return {str(key): item for key, item in value.items()}


def _only(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ReferenceResultsError("reference_unknown_field", path, unknown)


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ReferenceResultsError("reference_invalid_id", path)
    return value


def _text(value: Any, path: str, maximum: int = 2000) -> str:
    if not isinstance(value, str):
        raise ReferenceResultsError("reference_invalid_text", path)
    value = value.strip()
    if not value or len(value) > maximum or "\x00" in value:
        raise ReferenceResultsError("reference_invalid_text", path)
    return value


def _signature(value: Any, path: str) -> dict[str, str | int | float | bool]:
    row = _mapping(value, path)
    result: dict[str, str | int | float | bool] = {}
    for key, item in row.items():
        if not _ID_RE.fullmatch(key) or isinstance(item, (dict, list)) or item is None:
            raise ReferenceResultsError("reference_invalid_signature", f"{path}.{key}")
        if not isinstance(item, (str, int, float, bool)):
            raise ReferenceResultsError("reference_invalid_signature", f"{path}.{key}")
        if isinstance(item, str):
            if not item or len(item) > 256 or any(ord(character) < 32 for character in item):
                raise ReferenceResultsError("reference_invalid_signature", f"{path}.{key}")
        if isinstance(item, float) and not math.isfinite(item):
            raise ReferenceResultsError("reference_invalid_signature", f"{path}.{key}")
        result[key] = item
    return result


def validate_reference_snapshot(document: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(document, "")
    _only(
        root,
        {
            "schema", "schema_version", "version", "source_url", "source_asset", "captured_at",
            "title", "disclaimer", "result_sets",
        },
        "",
    )
    if root.get("schema") != REFERENCE_SCHEMA:
        raise ReferenceResultsError("reference_schema_mismatch", "schema")
    if root.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise ReferenceResultsError("reference_schema_version_unsupported", "schema_version")
    version = _text(root.get("version"), "version", 64)
    source_url = _text(root.get("source_url"), "source_url", 2048)
    parsed_url = urlparse(source_url)
    if parsed_url.scheme != "https" or parsed_url.netloc != "www.alphabrain-platform.com":
        raise ReferenceResultsError("reference_source_not_official", "source_url")
    captured_at = _text(root.get("captured_at"), "captured_at", 64)
    try:
        captured = datetime.fromisoformat(captured_at)
    except ValueError as exc:
        raise ReferenceResultsError("reference_invalid_capture_date", "captured_at") from exc
    if captured.tzinfo is None:
        raise ReferenceResultsError("reference_capture_timezone_required", "captured_at")
    source_asset = _text(root.get("source_asset"), "source_asset", 2048)
    if urlparse(source_asset).scheme != "https" or urlparse(source_asset).netloc != parsed_url.netloc:
        raise ReferenceResultsError("reference_source_not_official", "source_asset")

    raw_sets = root.get("result_sets")
    if not isinstance(raw_sets, list) or not raw_sets or len(raw_sets) > 256:
        raise ReferenceResultsError("reference_invalid_result_sets", "result_sets")
    set_ids: set[str] = set()
    normalized_sets: list[dict[str, Any]] = []
    for set_index, raw_set in enumerate(raw_sets):
        path = f"result_sets[{set_index}]"
        result_set = _mapping(raw_set, path)
        _only(
            result_set,
            {
                "id", "group", "category", "benchmark_id", "signature", "title",
                "kind", "metrics", "rows", "series", "notes",
            },
            path,
        )
        identifier = _identifier(result_set.get("id"), f"{path}.id")
        if identifier in set_ids:
            raise ReferenceResultsError("reference_duplicate_id", f"{path}.id", identifier)
        set_ids.add(identifier)
        kind = result_set.get("kind")
        if kind not in {"table", "series"}:
            raise ReferenceResultsError("reference_invalid_kind", f"{path}.kind")
        benchmark_id = _identifier(result_set.get("benchmark_id"), f"{path}.benchmark_id")
        signature = _signature(result_set.get("signature", {}), f"{path}.signature")
        raw_metrics = result_set.get("metrics")
        if not isinstance(raw_metrics, list) or not raw_metrics or len(raw_metrics) > 128:
            raise ReferenceResultsError("reference_invalid_metrics", f"{path}.metrics")
        metrics: list[dict[str, str]] = []
        metric_ids: set[str] = set()
        metric_units: dict[str, str] = {}
        for metric_index, raw_metric in enumerate(raw_metrics):
            metric_path = f"{path}.metrics[{metric_index}]"
            metric = _mapping(raw_metric, metric_path)
            _only(metric, {"id", "label", "unit", "direction"}, metric_path)
            metric_id = _identifier(metric.get("id"), f"{metric_path}.id")
            if metric_id in metric_ids:
                raise ReferenceResultsError("reference_duplicate_id", f"{metric_path}.id", metric_id)
            metric_ids.add(metric_id)
            unit = metric.get("unit")
            direction = metric.get("direction")
            if unit not in _UNITS or direction not in _DIRECTIONS:
                raise ReferenceResultsError("reference_invalid_metric_definition", metric_path)
            metrics.append(
                {
                    "id": metric_id,
                    "label": _text(metric.get("label"), f"{metric_path}.label", 256),
                    "unit": unit,
                    "direction": direction,
                }
            )
            metric_units[metric_id] = unit

        def values(raw_values: Any, value_path: str) -> dict[str, float | int | None]:
            mapping = _mapping(raw_values, value_path)
            unknown = sorted(set(mapping) - metric_ids)
            if unknown:
                raise ReferenceResultsError("reference_unknown_metric", value_path, unknown)
            result: dict[str, float | int | None] = {}
            for metric_id, item in mapping.items():
                if item is None:
                    result[metric_id] = None
                    continue
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise ReferenceResultsError("reference_invalid_metric_value", f"{value_path}.{metric_id}")
                numeric = float(item)
                if metric_units[metric_id] == "percent" and not 0 <= numeric <= 100:
                    raise ReferenceResultsError("reference_percent_out_of_range", f"{value_path}.{metric_id}")
                result[metric_id] = item
            return result

        normalized_rows: list[dict[str, Any]] = []
        normalized_series: list[dict[str, Any]] = []
        if kind == "table":
            raw_rows = result_set.get("rows")
            if not isinstance(raw_rows, list) or not raw_rows or result_set.get("series") not in (None, []):
                raise ReferenceResultsError("reference_invalid_rows", f"{path}.rows")
            row_ids: set[str] = set()
            for row_index, raw_row in enumerate(raw_rows):
                row_path = f"{path}.rows[{row_index}]"
                row = _mapping(raw_row, row_path)
                _only(row, {"id", "label", "origin", "values", "context"}, row_path)
                row_id = _identifier(row.get("id"), f"{row_path}.id")
                if row_id in row_ids:
                    raise ReferenceResultsError("reference_duplicate_id", f"{row_path}.id", row_id)
                row_ids.add(row_id)
                normalized_rows.append(
                    {
                        "id": row_id,
                        "label": _text(row.get("label"), f"{row_path}.label", 512),
                        "origin": _text(row.get("origin"), f"{row_path}.origin", 128),
                        "values": values(row.get("values", {}), f"{row_path}.values"),
                        "context": _signature(row.get("context", {}), f"{row_path}.context"),
                    }
                )
        else:
            raw_series = result_set.get("series")
            if not isinstance(raw_series, list) or not raw_series or result_set.get("rows") not in (None, []):
                raise ReferenceResultsError("reference_invalid_series", f"{path}.series")
            series_ids: set[str] = set()
            for series_index, raw_item in enumerate(raw_series):
                series_path = f"{path}.series[{series_index}]"
                item = _mapping(raw_item, series_path)
                _only(item, {"id", "label", "origin", "points", "context"}, series_path)
                series_id = _identifier(item.get("id"), f"{series_path}.id")
                if series_id in series_ids:
                    raise ReferenceResultsError("reference_duplicate_id", f"{series_path}.id", series_id)
                series_ids.add(series_id)
                raw_points = item.get("points")
                if not isinstance(raw_points, list) or not raw_points or len(raw_points) > 10000:
                    raise ReferenceResultsError("reference_invalid_points", f"{series_path}.points")
                points: list[dict[str, Any]] = []
                previous_step: float | None = None
                for point_index, raw_point in enumerate(raw_points):
                    point_path = f"{series_path}.points[{point_index}]"
                    point = _mapping(raw_point, point_path)
                    _only(point, {"step", "values"}, point_path)
                    step = point.get("step")
                    if (
                        isinstance(step, bool)
                        or not isinstance(step, (int, float))
                        or (previous_step is not None and step <= previous_step)
                    ):
                        raise ReferenceResultsError("reference_invalid_step", f"{point_path}.step")
                    previous_step = float(step)
                    points.append({"step": step, "values": values(point.get("values", {}), f"{point_path}.values")})
                normalized_series.append(
                    {
                        "id": series_id,
                        "label": _text(item.get("label"), f"{series_path}.label", 512),
                        "origin": _text(item.get("origin"), f"{series_path}.origin", 128),
                        "points": points,
                        "context": _signature(item.get("context", {}), f"{series_path}.context"),
                    }
                )
        normalized_sets.append(
            {
                "id": identifier,
                "group": _identifier(result_set.get("group"), f"{path}.group"),
                "category": _identifier(result_set.get("category"), f"{path}.category"),
                "benchmark_id": benchmark_id,
                "signature": signature,
                "title": _text(result_set.get("title"), f"{path}.title", 512),
                "kind": kind,
                "metrics": metrics,
                "rows": normalized_rows,
                "series": normalized_series,
                "notes": _text(
                    result_set.get(
                        "notes",
                        "Website display value; not independently reproduced.",
                    ),
                    f"{path}.notes",
                    4000,
                ),
            }
        )
    return {
        "schema": REFERENCE_SCHEMA,
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "version": version,
        "source_url": source_url,
        "source_asset": source_asset,
        "captured_at": captured_at,
        "title": _text(root.get("title"), "title", 512),
        "disclaimer": _text(root.get("disclaimer"), "disclaimer", 4000),
        "result_sets": normalized_sets,
    }


@lru_cache(maxsize=1)
def _cached_snapshot() -> dict[str, Any]:
    try:
        raw = yaml.safe_load(REFERENCE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ReferenceResultsError("reference_snapshot_unavailable", str(REFERENCE_SNAPSHOT_PATH)) from exc
    return validate_reference_snapshot(raw)


def load_reference_snapshot() -> dict[str, Any]:
    """Return an isolated copy of the packaged snapshot; never access the network."""

    return deepcopy(_cached_snapshot())


def signature_matches(reference: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    """Require every versioned reference discriminator to match exactly.

    The local signature may contain extra runtime metadata, but it cannot omit
    or disagree with any discriminator captured in the reference set.
    """

    return all(key in actual and actual[key] == value for key, value in reference.items())


def match_reference_results(
    benchmark_id: str,
    signature: Mapping[str, Any],
    *,
    snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_benchmark_id = _identifier(benchmark_id, "benchmark_id")
    normalized_signature = _signature(signature, "signature")
    if len(normalized_signature) > 64:
        raise ReferenceResultsError("reference_invalid_signature", "signature")
    validated = validate_reference_snapshot(snapshot) if snapshot is not None else load_reference_snapshot()
    matches = [
        deepcopy(result_set)
        for result_set in validated["result_sets"]
        if result_set["benchmark_id"] == normalized_benchmark_id
        and signature_matches(result_set["signature"], normalized_signature)
    ]
    return {
        "schema": "alphabrain.reference-results-match",
        "schema_version": 1,
        "snapshot": {
            key: validated[key]
            for key in ("version", "source_url", "source_asset", "captured_at", "title", "disclaimer")
        },
        "benchmark_id": normalized_benchmark_id,
        "signature": json.loads(json.dumps(normalized_signature, ensure_ascii=False)),
        "items": matches,
    }
