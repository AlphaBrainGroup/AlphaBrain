from __future__ import annotations

from copy import deepcopy

from alphabrain_ui.template_fingerprints import (
    canonical_template_spec,
    normalize_template_name,
    template_spec_fingerprint,
)


def _spec() -> dict:
    return {
        "architecture": {"backbone": "pi0_5", "action_head": "flow_matching"},
        "training": {"method": "sft", "checkpoint": "/models/pi05"},
        "dataset": {"id": "libero", "root": "/datasets/libero"},
        "parameters": {
            "run_id": "first-run",
            "output_root_dir": "/results/first-run",
            "learning_rate": 1e-4,
            "max_train_steps": 1000,
        },
        "resources": {
            "allocation": "fixed",
            "num_gpus": 2,
            "gpu_ids": [2, 3],
            "main_process_port": 29501,
        },
        "wandb": {
            "project": "AlphaBrain",
            "run_name": "first-run",
            "notes": "first note",
            "categories": ["videos", "metrics", "metrics"],
        },
        "metadata": {"description": "first description", "source_template_id": "source-1"},
        "expert_overrides": {"trainer.precision": "bf16"},
    }


def test_fingerprint_ignores_per_run_metadata_and_allocation_ids() -> None:
    first = _spec()
    second = deepcopy(first)
    second["parameters"].update(run_id="second-run", output_root_dir="/results/second-run")
    second["resources"].update(gpu_ids=[6, 7], main_process_port=29666)
    second["wandb"].update(run_name="second-run", notes="second note")
    second["wandb"]["categories"] = ["metrics", "videos"]
    second["metadata"] = {"description": "other", "source_template_id": "source-2"}

    assert template_spec_fingerprint(first) == template_spec_fingerprint(second)
    canonical = canonical_template_spec(second)
    assert canonical["resources"] == {"allocation": "auto", "num_gpus": 2}
    assert "metadata" not in canonical
    assert canonical["wandb"]["categories"] == ["metrics", "videos"]


def test_fingerprint_keeps_scientific_and_operational_choices() -> None:
    base = _spec()
    variants = []
    for path, value in (
        (("dataset", "id"), "bridge"),
        (("parameters", "learning_rate"), 2e-4),
        (("training", "checkpoint"), "/models/pi05-v2"),
        (("resources", "num_gpus"), 4),
        (("expert_overrides", "trainer.precision"), "fp32"),
    ):
        variant = deepcopy(base)
        variant[path[0]][path[1]] = value
        variants.append(variant)

    baseline = template_spec_fingerprint(base)
    assert all(template_spec_fingerprint(item) != baseline for item in variants)


def test_template_name_normalization_handles_unicode_space_and_case() -> None:
    assert normalize_template_name("  ＰＩ０ Template  ") == normalize_template_name("pi0 template")
