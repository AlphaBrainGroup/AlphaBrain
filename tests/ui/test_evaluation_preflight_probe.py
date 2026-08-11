from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import alphabrain_ui.evaluation_preflight as evaluation_preflight


def test_benchmark_probe_loads_real_entrypoint_with_repo_and_sibling_imports(
    tmp_path: Path,
) -> None:
    client_dir = tmp_path / "benchmark/eval"
    client_dir.mkdir(parents=True)
    simulator_root = tmp_path / "LIBERO"
    simulator_root.mkdir()
    (tmp_path / "repo_helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (client_dir / "adapter_helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    (simulator_root / "simulator_helper.py").write_text("VALUE = 3\n", encoding="utf-8")
    entrypoint = client_dir / "client.py"
    entrypoint.write_text(
        "from dataclasses import dataclass\n"
        "from repo_helper import VALUE as REPO_VALUE\n"
        "from adapter_helper import VALUE as ADAPTER_VALUE\n"
        "from simulator_helper import VALUE as SIMULATOR_VALUE\n"
        "@dataclass\n"
        "class ImportedClient:\n"
        "    value: int = REPO_VALUE + ADAPTER_VALUE + SIMULATOR_VALUE\n",
        encoding="utf-8",
    )

    evaluation_preflight._probe_cache.clear()
    result = evaluation_preflight.probe_benchmark_python(
        sys.executable,
        ("json",),
        tmp_path,
        {"LIBERO_HOME": str(simulator_root)},
        entrypoint=entrypoint,
        pythonpath_entries=(simulator_root,),
    )

    assert result == {"ok": True, "errors": {}}
    assert not list(tmp_path.rglob("__pycache__"))


def test_benchmark_probe_uses_runner_pythonpath_and_never_reflects_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    entrypoint = tmp_path / "benchmark/eval/client.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("raise RuntimeError('not executed by fake subprocess')\n", encoding="utf-8")
    libero_home = tmp_path / "libero-secret-path"
    existing_path = tmp_path / "existing-path"
    secret = "hf_super_secret_value"
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((command, kwargs))
        if len(calls) == 1:
            return SimpleNamespace(
                returncode=1,
                stdout=(
                    f"third-party output contains {secret}\n"
                    f'__ALPHABRAIN_BENCHMARK_PROBE__={{"entrypoint":"{secret}"}}\n'
                ),
                stderr=f"third-party stderr contains {secret}",
            )
        return SimpleNamespace(
            returncode=2,
            stdout=f"unstructured output contains {secret}\n",
            stderr=secret,
        )

    evaluation_preflight._probe_cache.clear()
    monkeypatch.setattr(evaluation_preflight.subprocess, "run", fake_run)
    first = evaluation_preflight.probe_benchmark_python(
        sys.executable,
        ("libero",),
        tmp_path,
        {
            "HF_TOKEN": secret,
            "LIBERO_HOME": str(libero_home),
            "PYTHONPATH": str(existing_path),
        },
        entrypoint=entrypoint,
        pythonpath_entries=(libero_home,),
    )
    second = evaluation_preflight.probe_benchmark_python(
        sys.executable,
        ("libero",),
        tmp_path,
        {
            "HF_TOKEN": secret + "-changed",
            "LIBERO_HOME": str(libero_home),
            "PYTHONPATH": str(existing_path),
        },
        entrypoint=entrypoint,
        pythonpath_entries=(libero_home,),
    )

    probe_env = calls[0][1]["env"]
    pythonpath = probe_env["PYTHONPATH"].split(os.pathsep)
    assert pythonpath[:3] == [
        str(tmp_path.resolve()),
        str(entrypoint.parent.resolve()),
        str(libero_home.resolve()),
    ]
    assert str(existing_path) in pythonpath
    assert probe_env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert calls[0][0][:2] == [sys.executable, "-c"]
    assert "shell" not in calls[0][1]
    assert first == {"ok": False, "errors": {"entrypoint": "ImportFailed"}}
    assert second == {"ok": False, "errors": {"python": "ProbeResultMissing"}}
    assert secret not in json.dumps(first)
    assert secret not in json.dumps(second)
    assert secret not in repr(evaluation_preflight._probe_cache)
    assert len(calls) == 2


def test_benchmark_probe_rejects_entrypoint_outside_repository(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")

    evaluation_preflight._probe_cache.clear()
    result = evaluation_preflight.probe_benchmark_python(
        sys.executable,
        ("json",),
        repo_root,
        entrypoint=outside,
    )

    assert result == {"ok": False, "errors": {"entrypoint": "ValueError"}}
