from alphabrain_ui.evaluation_registry import (
    get_evaluation_catalog,
    resolve_benchmark_parameters,
)


def test_libero_trials_are_bounded_by_available_initial_states() -> None:
    for benchmark_id in ("libero", "libero_plus"):
        _resolved, issues = resolve_benchmark_parameters(
            benchmark_id,
            "custom",
            {"num_trials": 51},
            include_experimental=True,
        )
        assert any(
            item["code"] == "benchmark_parameter_maximum"
            and item["field"] == "parameters.num_trials"
            for item in issues
        )


def test_libero_full_preset_uses_supported_trial_count() -> None:
    for benchmark_id in ("libero", "libero_plus"):
        resolved, issues = resolve_benchmark_parameters(
            benchmark_id,
            "full",
            include_experimental=True,
        )
        assert not issues
        assert resolved["num_trials"] == 50


def test_libero_catalog_exposes_the_same_trial_limit() -> None:
    catalog = get_evaluation_catalog(include_experimental=True)
    benchmarks = {item["id"]: item for item in catalog["benchmarks"]}
    for benchmark_id in ("libero", "libero_plus"):
        assert (
            benchmarks[benchmark_id]["parameter_schema"]["properties"]["num_trials"]["maximum"]
            == 50
        )
