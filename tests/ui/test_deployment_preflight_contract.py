from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

import alphabrain_ui.deployment_preflight as deployment_preflight
from alphabrain_ui.database import Checkpoint, Database, Experiment, ModelDeployment, User
from alphabrain_ui.gpu import GPUInfo
from alphabrain_ui.schemas import DeploymentRequest

REPO_ROOT = Path(__file__).resolve().parents[2]


class GPUFixture:
    available = True
    error = ""

    def __init__(self, *, count: int = 1, available: bool = True):
        self.count = count
        self.gpu_available = available

    def snapshot(self, reservations=None):  # type: ignore[no-untyped-def]
        reservations = reservations or {}
        return [
            GPUInfo(
                index=index,
                uuid=f"GPU-{index}",
                name=f"GPU {index}",
                memory_total_bytes=24 * 1024**3,
                memory_used_bytes=0,
                memory_free_bytes=24 * 1024**3,
                utilization_percent=0,
                temperature_c=30,
                processes=[],
                reserved_by_job_id=reservations.get(index),
                available=self.gpu_available and index not in reservations,
            )
            for index in range(self.count)
        ]


def codes(result: dict[str, Any]) -> set[str]:
    return {item["code"] for item in result["issues"]}


def make_checkpoint(path: Path, backbone: str = "qwen2_5_vl") -> Path:
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"static-preflight-must-not-read-this")
    (path / "framework_config.yaml").write_text(
        yaml.safe_dump({"framework": {"name": "QwenOFT", "qwenvl": {}}, "trainer": {}}),
        encoding="utf-8",
    )
    (path / "dataset_statistics.json").write_text("{}", encoding="utf-8")
    embedded = path / "vlm_pretrained"
    embedded.mkdir()
    (embedded / "config.json").write_text(json.dumps({"model_type": backbone}), encoding="utf-8")
    (embedded / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    return path


def make_database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "deployment-preflight.sqlite3")
    database.create_all()
    return database


def runtime() -> SimpleNamespace:
    return SimpleNamespace(repo_root=REPO_ROOT, host="127.0.0.1")


def request(
    checkpoint_source: dict[str, Any],
    *,
    port: int | None = None,
    idle_timeout_seconds: int = 1800,
    parameters: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
    acknowledge_experimental: bool = False,
    combination_id: str = "deploy_qwen2_5_oft",
) -> DeploymentRequest:
    endpoint: dict[str, Any] = {
        "scope": "local",
        "advertised_host": "127.0.0.1",
        "idle_timeout_seconds": idle_timeout_seconds,
    }
    if port is not None:
        endpoint["port"] = port
    return DeploymentRequest.model_validate(
        {
            "name": "deployment",
            "checkpoint_source": checkpoint_source,
            "combination_id": combination_id,
            "resources": resources or {"strategy": "auto", "gpu_count": 1, "gpu_ids": []},
            "endpoint": endpoint,
            "parameters": parameters or {"precision": "fp32"},
            "acknowledge_experimental": acknowledge_experimental,
        }
    )


def seed_indexed_checkpoint(database: Database, checkpoint_path: Path) -> tuple[str, str]:
    with database.session() as db:
        owner = User(username="owner", display_name="Owner", role="administrator")
        db.add(owner)
        db.flush()
        experiment = Experiment(
            owner_id=owner.id,
            name="training",
            family="imitation_learning",
            status="completed",
            spec={},
            resolved={},
        )
        db.add(experiment)
        db.flush()
        checkpoint = Checkpoint(
            experiment_id=experiment.id,
            path=str(checkpoint_path.resolve()),
            name=checkpoint_path.name,
            is_complete=True,
            is_resumable=False,
        )
        db.add(checkpoint)
        db.flush()
        return owner.id, checkpoint.id


def resolved_stub(_source: Any = None, **kwargs: Any) -> dict[str, Any]:
    parameters = dict(kwargs.get("parameters") or {})
    return {
        "valid": True,
        "command": ["python", "server.py"],
        "environment": {},
        "startup_timeout_seconds": 900,
        "checkpoint_path": "/models/checkpoint",
        "adapter_id": "base_framework_websocket",
        "combination_id": kwargs.get("combination_id"),
        "backbone_id": "qwen2_5_vl",
        "action_head_id": "mlp_regression",
        "resolved_parameters": parameters,
        "issues": [],
    }


def test_source_discriminator_is_authoritative_and_endpoint_overrides_parameters(
    tmp_path: Path, monkeypatch
) -> None:
    database = make_database(tmp_path)
    indexed_path = make_checkpoint(tmp_path / "indexed")
    local_path = make_checkpoint(tmp_path / "local")
    _owner_id, checkpoint_id = seed_indexed_checkpoint(database, indexed_path)
    monkeypatch.setattr(
        deployment_preflight,
        "probe_model_server_python",
        lambda *_args, **_kwargs: {"ok": True, "errors": {}},
    )
    monkeypatch.setattr(deployment_preflight, "_port_available", lambda *_args: True)

    indexed_payload = request(
        {"kind": "indexed", "checkpoint_id": checkpoint_id, "path": str(local_path)},
        port=15432,
        idle_timeout_seconds=77,
        parameters={"precision": "fp32", "port": 19999, "idle_timeout_seconds": 2},
    )
    local_payload = request(
        {"kind": "local", "checkpoint_id": checkpoint_id, "path": str(local_path)},
        port=15433,
        idle_timeout_seconds=78,
        parameters={"precision": "fp32", "port": 19998, "idle_timeout_seconds": 3},
    )
    with database.session() as db:
        indexed = deployment_preflight.validate_deployment_request(
            indexed_payload,
            db=db,
            runtime=runtime(),
            gpu_monitor=GPUFixture(),
            include_experimental=False,
        )
        local = deployment_preflight.validate_deployment_request(
            local_payload,
            db=db,
            runtime=runtime(),
            gpu_monitor=GPUFixture(),
            include_experimental=False,
        )

    assert indexed["ok"] is True, indexed
    assert indexed["resolved"]["checkpoint_id"] == checkpoint_id
    assert indexed["resolved"]["checkpoint_path"] == str(indexed_path.resolve())
    assert indexed["resolved"]["endpoint"]["port"] == 15432
    assert indexed["resolved"]["endpoint"]["idle_timeout_seconds"] == 77
    assert indexed["resolved"]["parameters"] == {"precision": "fp32"}
    assert indexed["resolved"]["command_preview"][-4:] == ["--port", "15432", "--idle_timeout", "77"]

    assert local["ok"] is True, local
    assert local["resolved"]["checkpoint_id"] is None
    assert local["resolved"]["checkpoint_path"] == str(local_path.resolve())
    assert local["resolved"]["endpoint"]["port"] == 15433
    assert local["resolved"]["endpoint"]["idle_timeout_seconds"] == 78
    assert local["resolved"]["parameters"] == {"precision": "fp32"}


def test_missing_or_incomplete_indexed_checkpoint_is_blocking(tmp_path: Path, monkeypatch) -> None:
    database = make_database(tmp_path)
    checkpoint_path = make_checkpoint(tmp_path / "checkpoint")
    _owner_id, checkpoint_id = seed_indexed_checkpoint(database, checkpoint_path)
    with database.session() as db:
        db.get(Checkpoint, checkpoint_id).is_complete = False
    monkeypatch.setattr(
        deployment_preflight,
        "probe_model_server_python",
        lambda *_args, **_kwargs: {"ok": True, "errors": {}},
    )

    with database.session() as db:
        incomplete = deployment_preflight.validate_deployment_request(
            request({"kind": "indexed", "checkpoint_id": checkpoint_id}),
            db=db,
            runtime=runtime(),
            gpu_monitor=GPUFixture(),
            include_experimental=False,
        )
        missing = deployment_preflight.validate_deployment_request(
            request({"kind": "indexed", "checkpoint_id": "missing"}),
            db=db,
            runtime=runtime(),
            gpu_monitor=GPUFixture(),
            include_experimental=False,
        )

    assert incomplete["ok"] is False
    assert "checkpoint_incomplete" in codes(incomplete)
    assert missing["ok"] is False
    assert "checkpoint_index_not_found" in codes(missing)


def test_experimental_combination_needs_both_gate_and_acknowledgement(tmp_path: Path, monkeypatch) -> None:
    database = make_database(tmp_path)
    flags: list[bool] = []

    def resolve(_source: Any = None, **kwargs: Any) -> dict[str, Any]:
        flags.append(bool(kwargs["include_experimental"]))
        return resolved_stub(**kwargs)

    monkeypatch.setattr(deployment_preflight, "resolve_deployment", resolve)
    monkeypatch.setattr(
        deployment_preflight,
        "get_deployment_catalog",
        lambda **_kwargs: {
            "deployment_combinations": [{"id": "experimental_combo", "status": "experimental"}]
        },
    )
    monkeypatch.setattr(
        deployment_preflight,
        "probe_model_server_python",
        lambda *_args, **_kwargs: {"ok": True, "errors": {}},
    )
    source = {"kind": "local", "path": str(tmp_path / "checkpoint")}
    with database.session() as db:
        disabled = deployment_preflight.validate_deployment_request(
            request(source, combination_id="experimental_combo"),
            db=db,
            runtime=runtime(),
            gpu_monitor=GPUFixture(),
            include_experimental=False,
        )
        unacknowledged = deployment_preflight.validate_deployment_request(
            request(source, combination_id="experimental_combo"),
            db=db,
            runtime=runtime(),
            gpu_monitor=GPUFixture(),
            include_experimental=True,
        )
        accepted = deployment_preflight.validate_deployment_request(
            request(source, combination_id="experimental_combo", acknowledge_experimental=True),
            db=db,
            runtime=runtime(),
            gpu_monitor=GPUFixture(),
            include_experimental=True,
        )

    assert "experimental_deployment_not_enabled" in codes(disabled)
    assert "experimental_risk_not_acknowledged" in codes(unacknowledged)
    assert accepted["ok"] is True
    assert flags == [False, True, True]


def test_gpu_queue_capacity_and_duplicate_port_checks_are_stable(tmp_path: Path, monkeypatch) -> None:
    database = make_database(tmp_path)
    monkeypatch.setattr(deployment_preflight, "resolve_deployment", resolved_stub)
    monkeypatch.setattr(
        deployment_preflight,
        "probe_model_server_python",
        lambda *_args, **_kwargs: {"ok": True, "errors": {}},
    )
    monkeypatch.setattr(
        deployment_preflight,
        "_port_available",
        lambda *_args: (_ for _ in ()).throw(AssertionError("reserved ports must not be probed")),
    )
    with database.session() as db:
        owner = User(username="port-owner", display_name="Owner")
        db.add(owner)
        db.flush()
        for index in range(2):
            db.add(
                ModelDeployment(
                    owner_id=owner.id,
                    name=f"reserved-{index}",
                    checkpoint_path="/models/checkpoint",
                    combination_id="deploy_qwen2_5_oft",
                    adapter_id="base_framework_websocket",
                    backbone_id="qwen2_5_vl",
                    action_head_id="mlp_regression",
                    status="queued",
                    requested_gpu_count=1,
                    requested_gpu_ids=[],
                    assigned_gpu_ids=[],
                    parameters={},
                    port=16661,
                    api_key_hash=str(index) * 64,
                    api_key_prefix=f"ab_{index}",
                )
            )

    fixed_busy = request(
        {"kind": "local", "path": str(tmp_path / "checkpoint")},
        port=16661,
        resources={"strategy": "fixed", "gpu_count": 1, "gpu_ids": [0]},
    )
    auto_too_large = request(
        {"kind": "local", "path": str(tmp_path / "checkpoint")},
        resources={"strategy": "auto", "gpu_count": 2, "gpu_ids": []},
    )
    missing_fixed = request(
        {"kind": "local", "path": str(tmp_path / "checkpoint")},
        resources={"strategy": "fixed", "gpu_count": 1, "gpu_ids": [9]},
    )
    with database.session() as db:
        fixed = deployment_preflight.validate_deployment_request(
            fixed_busy,
            db=db,
            runtime=runtime(),
            gpu_monitor=GPUFixture(available=False),
            include_experimental=False,
        )
        auto = deployment_preflight.validate_deployment_request(
            auto_too_large,
            db=db,
            runtime=runtime(),
            gpu_monitor=GPUFixture(count=1, available=False),
            include_experimental=False,
        )
        missing = deployment_preflight.validate_deployment_request(
            missing_fixed,
            db=db,
            runtime=runtime(),
            gpu_monitor=GPUFixture(),
            include_experimental=False,
        )

    assert {"deployment_port_reserved", "deployment_will_queue_for_gpu"} <= codes(fixed)
    assert {"insufficient_visible_gpus", "deployment_will_queue_for_gpu"} <= codes(auto)
    assert "requested_gpus_unavailable" in codes(missing)


def test_python_probe_never_deserializes_weights_or_reflects_output_secrets(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    secret = "hf_super_secret_value"

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((command, kwargs))
        if len(calls) == 1:
            return SimpleNamespace(
                returncode=1,
                stdout=(
                    "third-party noise\n"
                    '__ALPHABRAIN_MODEL_SERVER_PROBE__={"torch":"hf_super_secret_value"}\n'
                ),
                stderr=secret,
            )
        return SimpleNamespace(returncode=2, stdout="", stderr=secret)

    deployment_preflight._probe_cache.clear()
    monkeypatch.setattr(deployment_preflight.subprocess, "run", fake_run)
    first = deployment_preflight.probe_model_server_python(
        sys.executable,
        tmp_path,
        {"HF_TOKEN": secret},
    )
    second = deployment_preflight.probe_model_server_python(
        sys.executable,
        tmp_path,
        {"HF_TOKEN": secret + "-changed"},
    )

    script = calls[0][0][2]
    assert calls[0][0][:2] == [sys.executable, "-c"]
    assert "shell" not in calls[0][1]
    assert all(token not in script for token in ("torch.load", "load_file", "from_pretrained", "checkpoint"))
    assert first == {"ok": False, "errors": {"torch": "ImportFailed"}}
    assert second == {"ok": False, "errors": {"python": "ProbeResultMissing"}}
    assert secret not in json.dumps(first)
    assert secret not in json.dumps(second)
    assert secret not in repr(deployment_preflight._probe_cache)
    assert len(calls) == 2  # environment fingerprint prevents a stale cached result
