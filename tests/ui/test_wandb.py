from __future__ import annotations

import importlib.util
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from alphabrain_ui.configuration import ExperimentConfigurationError, resolve_experiment
from alphabrain_ui.launchers import build_launch_plan
from alphabrain_ui.schemas import ExperimentRequest, TemplateCreate, WandbAPIKeyUpdate, WandbRunConfig
from alphabrain_ui.wandb import WandbSecretStore, wandb_category_catalog, wandb_requires_api_key

REPO_ROOT = Path(__file__).resolve().parents[2]
_WAND_B_ADAPTER_SPEC = importlib.util.spec_from_file_location(
    "alphabrain_wandb_integration",
    REPO_ROOT / "AlphaBrain/training/trainer_utils/wandb_integration.py",
)
assert _WAND_B_ADAPTER_SPEC and _WAND_B_ADAPTER_SPEC.loader
_WAND_B_ADAPTER = importlib.util.module_from_spec(_WAND_B_ADAPTER_SPEC)
_WAND_B_ADAPTER_SPEC.loader.exec_module(_WAND_B_ADAPTER)
configure_wandb_module = _WAND_B_ADAPTER.configure_wandb_module


def experiment_spec() -> dict:
    return {
        "architecture": {"backbone": "qwen2_5_vl", "action_head": "mlp_regression"},
        "training": {"method": "imitation_learning"},
        "dataset": {"id": "libero", "mix": "libero_goal"},
        "resources": {"allocation": "auto", "num_gpus": 1},
        "parameters": {"run_id": "wandb-test"},
        "expert_overrides": {},
    }


def test_secret_store_is_atomic_owner_only_and_never_returns_a_prefix(tmp_path: Path) -> None:
    store = WandbSecretStore(tmp_path)
    assert store.status() == {"configured": False}

    first = "first-test-key-123456"
    assert store.set_api_key(first) == {"configured": True}
    first_inode = store.path.stat().st_ino
    assert store.read_api_key() == first
    assert stat.S_IMODE(store.secrets_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert store.status() == {"configured": True}
    assert "prefix" not in store.status()

    second = "second-test-key-654321"
    store.set_api_key(second)
    assert store.path.stat().st_ino != first_inode
    assert store.read_api_key() == second
    assert not list(store.secrets_dir.glob(".wandb-key-*"))
    assert store.delete_api_key() == {"configured": False}
    assert store.delete_api_key() == {"configured": False}


def test_secret_store_rejects_insecure_files_and_bad_keys(tmp_path: Path) -> None:
    store = WandbSecretStore(tmp_path)
    with pytest.raises(ValueError):
        store.set_api_key("has whitespace")
    store.set_api_key("valid-test-key-123")
    store.path.chmod(0o644)
    with pytest.raises(PermissionError):
        store.read_api_key()


def test_wandb_schemas_are_credential_free() -> None:
    parsed = WandbRunConfig.model_validate(
        {
            "enabled": True,
            "project": " robotics ",
            "tags": ["baseline", "baseline", "libero"],
            "categories": ["metrics", "config", "metrics"],
        }
    )
    assert parsed.project == "robotics"
    assert parsed.tags == ["baseline", "libero"]
    assert parsed.categories == ["metrics", "config"]
    assert WandbAPIKeyUpdate(api_key=" key-for-testing ").api_key == "key-for-testing"
    with pytest.raises(ValidationError):
        WandbRunConfig(enabled=True, project="   ")

    spec = experiment_spec()
    spec["wandb"] = {"enabled": True, "api_key": "must-not-persist"}
    with pytest.raises(ValidationError):
        ExperimentRequest(name="unsafe", spec=spec)
    with pytest.raises(ValidationError):
        TemplateCreate(name="unsafe", spec=spec)


def test_resolver_maps_metadata_and_selected_content_without_a_secret(tmp_path: Path) -> None:
    spec = experiment_spec()
    spec["wandb"] = {
        "enabled": True,
        "mode": "online",
        "project": "robotics",
        "entity": "lab",
        "run_name": "qwen-baseline",
        "group": "libero-ablation",
        "job_type": "finetune",
        "tags": ["qwen", "libero"],
        "notes": "UI-managed run",
        "categories": ["metrics", "config"],
    }
    resolved = resolve_experiment(spec, REPO_ROOT, environment={})
    serialized = str(resolved)
    assert "api_key" not in serialized.lower()
    assert resolved["resolved_config"]["wandb_project"] == "robotics"
    assert resolved["resolved_config"]["wandb_upload_categories"] == ["metrics", "config"]
    environment = resolved["command_preview"]["environment"]
    assert environment["WANDB_PROJECT"] == "robotics"
    assert environment["WANDB_NAME"] == "qwen-baseline"
    assert environment["WANDB__DISABLE_STATS"] == "true"
    assert environment["WANDB_CONFIG_PATHS"] == "${RESOLVED_CONFIG_PATH}"
    assert "WANDB_API_KEY" not in environment
    assert wandb_requires_api_key(resolved) is True

    stages = build_launch_plan(
        REPO_ROOT,
        tmp_path,
        "wandb-test",
        resolved,
        snapshot_key="wandb-test",
        write_snapshots=True,
    )
    assert stages[0].environment["WANDB_CONFIG_PATHS"] == stages[0].config_snapshot_path
    assert "WANDB_API_KEY" not in stages[0].environment


def test_resolver_rejects_categories_the_selected_trainer_does_not_emit() -> None:
    spec = experiment_spec()
    spec["wandb"] = {
        "enabled": True,
        "project": "robotics",
        "categories": ["checkpoints", "videos"],
    }
    with pytest.raises(ExperimentConfigurationError) as raised:
        resolve_experiment(spec, REPO_ROOT, environment={})
    assert raised.value.issues[0]["code"] == "wandb_category_not_supported"
    assert raised.value.issues[0]["detail"]["categories"] == ["checkpoints", "videos"]
    catalog = {item["id"]: item for item in wandb_category_catalog()}
    assert catalog["checkpoints"]["supported"] is False
    assert catalog["videos"]["supported"] is False


def test_trainer_adapter_overrides_metadata_and_filters_content(monkeypatch) -> None:
    calls: dict[str, list] = {"init": [], "log": []}

    def original_init(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["init"].append((args, kwargs))
        return object()

    def original_log(data, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls["log"].append((data, args, kwargs))
        return None

    fake = SimpleNamespace(init=original_init, log=original_log)
    configure_wandb_module(fake)
    configure_wandb_module(fake)
    monkeypatch.setenv("ALPHABRAIN_WANDB_CATEGORIES", "videos")
    monkeypatch.setenv("WANDB_PROJECT", "ui-project")
    monkeypatch.setenv("WANDB_NAME", "ui-run")

    fake.init(project="legacy", name="legacy", config={"batch_size": 4})
    fake.log({"loss": 1.0, "video/success": object()}, step=3)
    assert calls["init"][0][1] == {"project": "ui-project", "name": "ui-run"}
    assert calls["log"][0][0] == {"video/success": calls["log"][0][0]["video/success"]}
