from __future__ import annotations

import json
import stat
import time
from pathlib import Path

import httpx
import pytest

from scripts.agent.alphabrain_api import (
    AlphaBrainAPIClient,
    NetworkFailure,
    TransportSecurityFailure,
    UsageFailure,
    WorkloadFailure,
    _render_response,
    build_parser,
    is_dangerous,
    iter_sse,
    normalize_api_path,
    normalize_base_url,
    redact,
    run,
    validate_output_destination,
)


def json_response(status: int, payload: object, *, headers: list[tuple[str, str]] | None = None) -> httpx.Response:
    return httpx.Response(status, json=payload, headers=headers)


def personal_transport(handler):  # type: ignore[no-untyped-def]
    def wrapped(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/setup/status":
            return json_response(200, {"initialized": True, "deployment_mode": "personal"})
        return handler(request)

    return httpx.MockTransport(wrapped)


def test_path_safety_and_redaction() -> None:
    assert normalize_api_path("health") == "/api/v1/health"
    assert normalize_api_path("/api/openapi.json") == "/api/openapi.json"
    assert is_dangerous("DELETE", "/api/v1/templates/id") is True
    assert is_dangerous("POST", "/api/v1/jobs/id/force-kill") is True
    assert is_dangerous("POST", "/api/v1/deployments/id/restart?wait=true") is True
    assert is_dangerous("POST", "/api/v1/experiments/preflight") is False
    with pytest.raises(UsageFailure, match="without a path"):
        normalize_base_url("https://server.example/alphabrain")
    for bypass_path in ("/deployments/d/stop/.", "/deployments/d/%73top"):
        with pytest.raises(UsageFailure):
            normalize_api_path(bypass_path)
    assert redact(
        {
            "api_key": "secret",
            "api_key_prefix": "ab_12",
            "authorization": "Bearer secret",
            "download_token": "hf_secret",
            "nested": [{"password": "pw"}],
        }
    ) == {
        "api_key": "<redacted>",
        "api_key_prefix": "ab_12",
        "authorization": "<redacted>",
        "download_token": "<redacted>",
        "nested": [{"password": "<redacted>"}],
    }


def test_public_setup_does_not_require_login_or_csrf() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/api/v1/setup/status":
            return json_response(200, {"initialized": False, "deployment_mode": "lab"})
        if request.url.path == "/api/v1/setup":
            assert "X-CSRF-Token" not in request.headers
            return json_response(200, {"initialized": True})
        raise AssertionError(request.url)

    with AlphaBrainAPIClient(
        base_url="https://server.example",
        transport=httpx.MockTransport(handler),
    ) as client:
        client.setup_status()
        assert client.request("POST", "/setup", json={"mode": "personal"}).json()["initialized"] is True
    assert "/api/v1/auth/login" not in [request.url.path for request in calls]


@pytest.mark.parametrize("path", ["/setup", "/auth/login"])
def test_public_credential_routes_refuse_remote_plain_http(path: str) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("credential request must be rejected before transport")

    with AlphaBrainAPIClient(
        base_url="http://server.example:8000",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(TransportSecurityFailure):
            client.request("POST", path, json={"password": "password123"})


def test_personal_mutation_refuses_remote_plain_http() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("mutation must be rejected before transport")

    with AlphaBrainAPIClient(
        base_url="http://server.example:8000",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(TransportSecurityFailure, match="sensitive API requests"):
            client.request("POST", "/templates", json={"name": "test"})


def test_secret_output_refuses_remote_plain_http_even_for_get(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("secret response must be rejected before transport")

    with AlphaBrainAPIClient(
        base_url="http://server.example:8000",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(TransportSecurityFailure, match="secret responses"):
            client.request_value("GET", "/deployments/d1", secret_output=str(tmp_path / "secret.json"))


def test_explicit_insecure_http_override_allows_personal_mutation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/setup/status":
            return json_response(200, {"initialized": True, "deployment_mode": "personal"})
        assert request.method == "POST"
        assert request.url.path == "/api/v1/templates"
        return json_response(200, {"id": "template-id"})

    with AlphaBrainAPIClient(
        base_url="http://server.example:8000",
        allow_insecure_http=True,
        transport=httpx.MockTransport(handler),
    ) as client:
        assert client.request("POST", "/templates", json={"name": "test"}).json() == {"id": "template-id"}


def test_non_positive_request_timeout_is_rejected() -> None:
    with pytest.raises(UsageFailure, match="timeout must be positive"):
        AlphaBrainAPIClient(base_url="http://127.0.0.1:8000", timeout=0)


def test_client_ignores_inherited_proxy_environment_by_default(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ALL_PROXY", "socks5://proxy.invalid:1080")
    with AlphaBrainAPIClient(base_url="http://127.0.0.1:8000") as client:
        assert client.http._trust_env is False


def test_password_and_json_cannot_share_infer_stdin() -> None:
    args = build_parser().parse_args(
        ["--password-stdin", "infer", "deployment-1", "--instruction", "move", "--states", "-"]
    )
    with pytest.raises(UsageFailure, match="cannot share stdin"):
        run(args)


def test_request_outputs_are_exclusive_and_secret_output_must_be_new(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["request", "GET", "/health", "--output", "response.json", "--secret-output", "secret.json"]
        )
    existing = tmp_path / "existing-secret.json"
    existing.write_text("old secret\n", encoding="utf-8")
    with pytest.raises(UsageFailure, match="already exists"):
        validate_output_destination(str(existing), require_new=True)
    args = build_parser().parse_args(["request", "GET", "/health", "--overwrite"])
    with pytest.raises(UsageFailure, match="requires --output"):
        run(args)


def test_personal_status_and_protected_request_need_no_login() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/api/v1/health":
            return json_response(200, {"status": "ok", "version": "test"})
        if request.url.path == "/api/v1/gpus":
            return json_response(200, {"items": []})
        raise AssertionError(request.url)

    with AlphaBrainAPIClient(
        base_url="http://127.0.0.1:8000",
        transport=personal_transport(handler),
    ) as client:
        assert client.status()["health"]["status"] == "ok"
        assert client.request("GET", "/gpus").json() == {"items": []}
    assert "/api/v1/auth/login" not in seen


def test_lab_login_keeps_cookie_in_memory_and_adds_csrf() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/api/v1/setup/status":
            return json_response(200, {"initialized": True, "deployment_mode": "lab"})
        if request.url.path == "/api/v1/auth/login":
            assert json.loads(request.content) == {"username": "alice", "password": "password123"}
            return json_response(
                200,
                {"user": {"username": "alice"}, "csrf_token": "csrf-value"},
                headers=[
                    ("set-cookie", "alphabrain_session=session-value; Path=/; HttpOnly"),
                    ("set-cookie", "alphabrain_csrf=csrf-value; Path=/"),
                ],
            )
        if request.url.path == "/api/v1/templates":
            assert request.headers["X-CSRF-Token"] == "csrf-value"
            assert "alphabrain_session=session-value" in request.headers["cookie"]
            return json_response(200, {"id": "template-id"})
        raise AssertionError(request.url)

    with AlphaBrainAPIClient(
        base_url="https://server.example",
        username="alice",
        password_provider=lambda: "password123",
        transport=httpx.MockTransport(handler),
    ) as client:
        assert client.request("POST", "/templates", json={"name": "test"}).json()["id"] == "template-id"
    assert [request.url.path for request in calls].count("/api/v1/auth/login") == 1


def test_lab_refuses_password_over_remote_plain_http() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/setup/status":
            return json_response(200, {"initialized": True, "deployment_mode": "lab"})
        raise AssertionError("login must not be attempted over insecure HTTP")

    with AlphaBrainAPIClient(
        base_url="http://server.example:8000",
        username="alice",
        password_provider=lambda: "password123",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(TransportSecurityFailure):
            client.request("GET", "/gpus")


def test_live_openapi_operation_lookup() -> None:
    document = {
        "openapi": "3.1.0",
        "paths": {"/api/v1/gpus": {"get": {"operationId": "gpus_api_v1_gpus_get"}}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/openapi.json"
        return json_response(200, document)

    with AlphaBrainAPIClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.schema("GET", "/gpus")
    assert result["path"] == "/api/v1/gpus"
    assert result["operation"]["operationId"] == "gpus_api_v1_gpus_get"


def test_atomic_file_output_and_secret_permissions(tmp_path: Path) -> None:
    response = httpx.Response(
        200,
        json={"deployment": {"id": "d1"}, "api_key": "ab_secret"},
        headers={"content-type": "application/json"},
    )
    secret_path = tmp_path / "deployment-key.json"
    rendered = _render_response(response, output=None, secret_output=str(secret_path))
    assert rendered["api_key"] == "ab_secret"
    assert json.loads(secret_path.read_text())["api_key"] == "ab_secret"
    assert stat.S_IMODE(secret_path.stat().st_mode) == 0o600

    binary = httpx.Response(200, content=b"archive", headers={"content-type": "application/octet-stream"})
    with pytest.raises(UsageFailure, match="require --output"):
        _render_response(binary, output=None, secret_output=None)
    output = tmp_path / "archive.tar"
    assert _render_response(binary, output=str(output), secret_output=None)["size_bytes"] == 7
    assert output.read_bytes() == b"archive"


def test_sse_parser_preserves_ids_and_multiline_data() -> None:
    events = list(
        iter_sse(
            iter(
                [
                    "id: 3:4",
                    "event: progress",
                    'data: {"status":',
                    'data: "running"}',
                    "",
                ]
            )
        )
    )
    assert events == [{"id": "3:4", "event": "progress", "data": '{"status":\n"running"}'}]
    assert list(iter_sse(iter([": heartbeat", ""]))) == [{"comment": "heartbeat"}]


def test_watch_job_uses_sse_terminal_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/setup/status":
            return json_response(200, {"initialized": True, "deployment_mode": "personal"})
        if request.url.path == "/api/v1/jobs/job-1/events":
            return httpx.Response(
                200,
                text='id: 12:3\nevent: status\ndata: {"status":"completed"}\n\n',
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(request.url)

    with AlphaBrainAPIClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.watch(kind="job", item_id="job-1", timeout=1)
    assert result == {"id": "job-1", "kind": "job", "status": "completed"}


def test_watch_utility_polls_and_reports_failure() -> None:
    polls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        if request.url.path == "/api/v1/setup/status":
            return json_response(200, {"initialized": True, "deployment_mode": "personal"})
        if request.url.path == "/api/v1/utilities":
            polls += 1
            status = "queued" if polls == 1 else "failed"
            return json_response(200, [{"id": "utility-1", "status": status}])
        raise AssertionError(request.url)

    with AlphaBrainAPIClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(WorkloadFailure, match="unsuccessful status 'failed'"):
            client.watch(kind="utility", item_id="utility-1", timeout=1, poll_interval=0.01)
    assert polls == 2


@pytest.mark.parametrize("until", ["terminal", "failed"])
def test_watch_never_treats_failed_status_as_success(until: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/setup/status":
            return json_response(200, {"initialized": True, "deployment_mode": "personal"})
        if request.url.path == "/api/v1/jobs/job-1/events":
            return httpx.Response(
                200,
                text='event: status\ndata: {"status":"failed"}\n\n',
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(request.url)

    with AlphaBrainAPIClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(WorkloadFailure, match="unsuccessful status"):
            client.watch(kind="job", item_id="job-1", until=until, timeout=1)


def test_watch_timeout_advances_on_sse_heartbeats() -> None:
    class HeartbeatStream(httpx.SyncByteStream):
        def __iter__(self):  # type: ignore[no-untyped-def]
            for _index in range(50):
                time.sleep(0.01)
                yield b": heartbeat\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/setup/status":
            return json_response(200, {"initialized": True, "deployment_mode": "personal"})
        if request.url.path == "/api/v1/jobs/job-1/events":
            return httpx.Response(200, stream=HeartbeatStream(), headers={"content-type": "text/event-stream"})
        if request.url.path == "/api/v1/jobs/job-1":
            return json_response(200, {"id": "job-1", "status": "running"})
        raise AssertionError(request.url)

    started = time.monotonic()
    with AlphaBrainAPIClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(NetworkFailure, match="timed out"):
            client.watch(kind="job", item_id="job-1", timeout=0.05)
    assert time.monotonic() - started < 0.2


def test_download_streams_atomically_and_preserves_old_file_on_failure(tmp_path: Path) -> None:
    class ChunkStream(httpx.SyncByteStream):
        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail

        def __iter__(self):  # type: ignore[no-untyped-def]
            yield b"first-"
            if self.fail:
                raise httpx.ReadError("stream failed")
            yield b"second"

    failing = False

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/setup/status":
            return json_response(200, {"initialized": True, "deployment_mode": "personal"})
        if request.url.path == "/api/v1/utilities/run-1/output":
            return httpx.Response(
                200,
                stream=ChunkStream(fail=failing),
                headers={"content-type": "application/zip"},
            )
        raise AssertionError(request.url)

    output = tmp_path / "package.zip"
    with AlphaBrainAPIClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.download("GET", "/utilities/run-1/output", output=str(output))
        assert result["size_bytes"] == len(b"first-second")
        assert output.read_bytes() == b"first-second"

        failing = True
        with pytest.raises(NetworkFailure, match="download failed"):
            client.download("GET", "/utilities/run-1/output", output=str(output), overwrite=True)
    assert output.read_bytes() == b"first-second"
    assert list(tmp_path.glob(".package.zip.*.tmp")) == []


def test_download_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("existing output must be rejected before transport")

    output = tmp_path / "package.zip"
    output.write_bytes(b"keep-me")
    with AlphaBrainAPIClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(UsageFailure, match="already exists"):
            client.download("GET", "/utilities/run-1/output", output=str(output))
    assert calls == 0
    assert output.read_bytes() == b"keep-me"


def test_download_explicit_overwrite_replaces_existing_file(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/setup/status":
            return json_response(200, {"initialized": True, "deployment_mode": "personal"})
        if request.url.path == "/api/v1/utilities/run-1/output":
            return httpx.Response(200, content=b"replacement", headers={"content-type": "application/zip"})
        raise AssertionError(request.url)

    output = tmp_path / "package.zip"
    output.write_bytes(b"old")
    with AlphaBrainAPIClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.download("GET", "/utilities/run-1/output", output=str(output), overwrite=True)
    assert result["size_bytes"] == len(b"replacement")
    assert output.read_bytes() == b"replacement"


def test_binary_response_requires_output_without_reading_body() -> None:
    class ExplodingStream(httpx.SyncByteStream):
        def __iter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("binary body must not be buffered without --output")
            yield b"unreachable"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/setup/status":
            return json_response(200, {"initialized": True, "deployment_mode": "personal"})
        if request.url.path == "/api/v1/utilities/run-1/output":
            return httpx.Response(
                200,
                stream=ExplodingStream(),
                headers={"content-type": "application/zip"},
            )
        raise AssertionError(request.url)

    with AlphaBrainAPIClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(UsageFailure, match="require --output"):
            client.request_value("GET", "/utilities/run-1/output")


def test_infer_builds_multipart_and_keeps_save_inputs_opt_in(tmp_path: Path) -> None:
    image = tmp_path / "view.png"
    image.write_bytes(b"not-a-real-png")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/setup/status":
            return json_response(200, {"initialized": True, "deployment_mode": "personal"})
        if request.url.path == "/api/v1/deployments/deployment-1/infer":
            body = request.read()
            assert request.headers["content-type"].startswith("multipart/form-data;")
            assert b'pick up the block' in body
            assert b'name="save_inputs"' in body
            assert b"false" in body
            assert b'name="image_counts"' in body
            assert b"[1]" in body
            assert b'filename="view.png"' in body
            return json_response(200, {"id": "inference-1", "status": "completed"})
        raise AssertionError(request.url)

    with AlphaBrainAPIClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.infer(
            deployment_id="deployment-1",
            instructions=["pick up the block"],
            image_paths=[str(image)],
            states_path=None,
            save_inputs=False,
            image_counts=[1],
        )
    assert result["id"] == "inference-1"
