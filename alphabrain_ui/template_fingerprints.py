from __future__ import annotations

import hashlib
import json
import unicodedata
from copy import deepcopy
from typing import Any, Mapping

from .configuration import make_template_spec

TEMPLATE_FINGERPRINT_VERSION = 1


def normalize_template_name(value: str) -> str:
    """Normalize a display name for owner-scoped collision checks."""

    return unicodedata.normalize("NFKC", value).strip().casefold()


def canonical_template_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Remove per-run presentation/allocation fields before duplicate checks.

    The result deliberately remains conservative: scientific and operational
    choices stay in the identity, while names, provenance, fixed device IDs,
    and other values that change on every run do not.
    """

    value = make_template_spec(deepcopy(dict(spec)))
    value.pop("metadata", None)

    wandb = value.get("wandb")
    if isinstance(wandb, dict):
        wandb.pop("run_name", None)
        wandb.pop("notes", None)
        categories = wandb.get("categories")
        if isinstance(categories, list):
            wandb["categories"] = sorted({str(item) for item in categories})

    return value


def canonical_template_json(spec: Mapping[str, Any]) -> str:
    return json.dumps(
        canonical_template_spec(spec),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def template_spec_fingerprint(spec: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_template_json(spec).encode("utf-8")).hexdigest()
