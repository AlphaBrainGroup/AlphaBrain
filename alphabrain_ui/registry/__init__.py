"""Declarative capability-registry loader.

The registry is deliberately data-only.  UI and API code should consume it
through :mod:`alphabrain_ui.capabilities` instead of opening ``catalog.yaml``
directly, so filtering and compatibility semantics stay consistent.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

CATALOG_PATH = Path(__file__).with_name("catalog.yaml")


@lru_cache(maxsize=1)
def _cached_catalog() -> dict[str, Any]:
    with CATALOG_PATH.open("r", encoding="utf-8") as stream:
        catalog = yaml.safe_load(stream)
    if not isinstance(catalog, dict):
        raise RuntimeError(f"Capability registry must be a mapping: {CATALOG_PATH}")
    return catalog


def load_catalog() -> dict[str, Any]:
    """Return an isolated copy of the packaged capability catalog."""

    return deepcopy(_cached_catalog())


def catalog_for_runtime(
    catalog: Mapping[str, Any],
    *,
    demo_mode: bool,
) -> dict[str, Any]:
    """Return the catalog visible to the selected platform runtime.

    Entries marked ``availability: demo_only`` remain packaged so the isolated
    demo platform can exercise them, but are removed at the backend boundary
    for every regular platform API and launch path.
    """

    result = deepcopy(dict(catalog))
    if demo_mode:
        return result

    def available(row: Any) -> bool:
        return isinstance(row, Mapping) and row.get("availability") != "demo_only"

    components = result.setdefault("components", {})
    component_ids: dict[str, set[str]] = {}
    for category in ("backbones", "action_heads", "training_methods", "datasets"):
        rows = [row for row in components.get(category, []) if available(row)]
        components[category] = rows
        component_ids[category] = {
            str(row["id"]) for row in rows if isinstance(row.get("id"), str)
        }

    combinations = [
        row
        for row in result.get("combinations", [])
        if available(row)
        and row.get("backbone") in component_ids["backbones"]
        and row.get("action_head") in component_ids["action_heads"]
        and row.get("method") in component_ids["training_methods"]
        and set(row.get("datasets", [])) <= component_ids["datasets"]
    ]
    result["combinations"] = combinations
    combination_ids = {
        str(row["id"]) for row in combinations if isinstance(row.get("id"), str)
    }

    adapters = [row for row in result.get("deployment_adapters", []) if available(row)]
    result["deployment_adapters"] = adapters
    adapter_ids = {
        str(row["id"]) for row in adapters if isinstance(row.get("id"), str)
    }
    result["deployment_combinations"] = [
        row
        for row in result.get("deployment_combinations", [])
        if available(row)
        and row.get("backbone") in component_ids["backbones"]
        and row.get("action_head") in component_ids["action_heads"]
        and row.get("adapter") in adapter_ids
        and set(row.get("source_combinations", [])) <= combination_ids
    ]
    return result


__all__ = ["CATALOG_PATH", "catalog_for_runtime", "load_catalog"]
