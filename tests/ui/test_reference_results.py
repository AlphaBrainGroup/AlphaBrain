from __future__ import annotations

from copy import deepcopy

import pytest

from alphabrain_ui.reference_results import (
    REFERENCE_SCHEMA,
    ReferenceResultsError,
    load_reference_snapshot,
    match_reference_results,
    signature_matches,
    validate_reference_snapshot,
)


def test_packaged_reference_snapshot_has_explicit_source_date_schema_and_version() -> None:
    snapshot = load_reference_snapshot()
    assert snapshot["schema"] == REFERENCE_SCHEMA
    assert snapshot["schema_version"] == 1
    assert snapshot["version"] == "2026.07.16.1"
    assert snapshot["source_url"] == "https://www.alphabrain-platform.com/#features"
    assert snapshot["source_asset"].startswith("https://www.alphabrain-platform.com/assets/")
    assert snapshot["captured_at"] == "2026-07-16T13:08:00+08:00"
    assert {item["group"] for item in snapshot["result_sets"]} >= {
        "neurovla", "multiple_backbones", "world_model", "robocasa365", "rl_token"
    }
    assert len({item["id"] for item in snapshot["result_sets"]}) == len(snapshot["result_sets"])
    # Callers cannot mutate the process-wide cached copy.
    snapshot["version"] = "mutated"
    assert load_reference_snapshot()["version"] == "2026.07.16.1"


def test_reference_matching_requires_benchmark_and_every_versioned_signature_field() -> None:
    matching = match_reference_results(
        "libero",
        {"suite": "libero_all", "protocol": "website_backbone_table", "local_run_id": "extra-is-ok"},
    )
    assert [item["id"] for item in matching["items"]] == ["libero_backbone_reference"]
    assert matching["snapshot"]["source_url"].startswith("https://www.alphabrain-platform.com/")
    assert match_reference_results(
        "libero", {"suite": "libero_goal", "protocol": "website_backbone_table"}
    )["items"] == []
    assert match_reference_results(
        "robocasa365", {"suite": "libero_all", "protocol": "website_backbone_table"}
    )["items"] == []
    assert signature_matches({"suite": "libero_all"}, {"suite": "libero_all", "episodes": 50})
    assert not signature_matches({"suite": "libero_all"}, {"suite": "libero_goal"})


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda value: value.update(
                source_url="http://www.alphabrain-platform.com/#features"
            ),
            "reference_source_not_official",
        ),
        (lambda value: value.update(source_url="https://example.com/#features"), "reference_source_not_official"),
        (lambda value: value.update(captured_at="2026-07-16T13:08:00"), "reference_capture_timezone_required"),
        (lambda value: value.update(schema_version=99), "reference_schema_version_unsupported"),
        (
            lambda value: value["result_sets"][0]["rows"][0]["values"].update(libero_object=101),
            "reference_percent_out_of_range",
        ),
        (
            lambda value: value["result_sets"].append(deepcopy(value["result_sets"][0])),
            "reference_duplicate_id",
        ),
    ],
)
def test_reference_snapshot_validation_rejects_wrong_provenance_and_invalid_metrics(mutation, expected: str) -> None:
    snapshot = load_reference_snapshot()
    mutation(snapshot)
    with pytest.raises(ReferenceResultsError) as error:
        validate_reference_snapshot(snapshot)
    assert error.value.code == expected
