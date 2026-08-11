#!/usr/bin/env python3
"""Safe command-line transport for the AlphaBrain research-console API.

The backend remains the source of truth for schemas, authorization, preflight,
and workload state.  This helper only supplies deterministic authentication,
CSRF handling, redaction, streaming, and atomic file output for AI agents and
humans working without the browser frontend.
"""

from __future__ import annotations

import argparse
import getpass
import ipaddress
import json
import mimetypes
import os
import re
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_AUTH = 3
EXIT_HTTP = 4
EXIT_WORKLOAD = 5
EXIT_NETWORK = 6

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
DANGEROUS_ACTIONS = {"cancel", "stop", "terminate", "force-kill", "restart", "retry", "rotate-key"}
HARD_FAILURE_STATUSES = {"failed", "cancelled", "interrupted", "dependency_failed"}
PUBLIC_API_PATHS = {
    "/api/openapi.json",
    "/api/v1/auth/login",
    "/api/v1/health",
    "/api/v1/setup",
    "/api/v1/setup/status",
}
CSRF_EXEMPT_PATHS = {"/api/v1/auth/login", "/api/v1/setup"}
SECRET_KEY_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "controller_key",
    "cookie",
    "download_token",
    "password",
    "passwd",
    "private_key",
    "publish_token",
    "refresh_token",
    "secret",
    "session_token",
    "set_cookie",
    "token",
    "access_token",
    "auth_token",
    "csrf_token",
}


class ClientFailure(RuntimeError):
    """Base class for failures with a stable process exit code."""

    exit_code = EXIT_HTTP


class UsageFailure(ClientFailure):
    exit_code = EXIT_USAGE


class AuthenticationFailure(ClientFailure):
    exit_code = EXIT_AUTH


class TransportSecurityFailure(AuthenticationFailure):
    pass


class ResponseFailure(ClientFailure):
    def __init__(self, status_code: int, detail: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {_detail_message(detail)}")


class WorkloadFailure(ClientFailure):
    exit_code = EXIT_WORKLOAD


class NetworkFailure(ClientFailure):
    exit_code = EXIT_NETWORK


@dataclass(frozen=True)
class WatchSpec:
    detail_path: str | None
    events_path: str | None
    terminal_statuses: frozenset[str]
    default_success_status: str


WATCH_SPECS: dict[str, WatchSpec] = {
    "job": WatchSpec(
        detail_path="/jobs/{id}",
        events_path="/jobs/{id}/events",
        terminal_statuses=frozenset(
            {"completed", "failed", "stopped", "cancelled", "dependency_failed", "interrupted"}
        ),
        default_success_status="completed",
    ),
    "deployment": WatchSpec(
        detail_path="/deployments/{id}",
        events_path="/deployments/{id}/events",
        terminal_statuses=frozenset({"stopped", "failed", "cancelled"}),
        default_success_status="running",
    ),
    "evaluation": WatchSpec(
        detail_path="/evaluations/{id}",
        events_path="/evaluations/{id}/events",
        terminal_statuses=frozenset({"completed", "failed", "stopped", "cancelled", "interrupted"}),
        default_success_status="completed",
    ),
    "utility": WatchSpec(
        detail_path=None,
        events_path=None,
        terminal_statuses=frozenset({"completed", "failed", "cancelled", "stopped", "interrupted"}),
        default_success_status="completed",
    ),
}


def _detail_message(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, Mapping):
        for key in ("message", "code", "detail"):
            value = detail.get(key)
            if isinstance(value, str) and value:
                return value
    try:
        return json.dumps(redact(detail), ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(detail)


def _secret_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    if normalized in {"api_key_prefix", "token_prefix", "secret_configured"}:
        return False
    return normalized in SECRET_KEY_NAMES or normalized.endswith(
        ("_api_key", "_access_token", "_auth_token", "_secret", "_password")
    )


def redact(value: Any, *, parent_key: str = "") -> Any:
    """Return a JSON-compatible value with credential-shaped fields masked."""

    if parent_key and _secret_key(parent_key):
        return "<redacted>" if value not in (None, "") else value
    if isinstance(value, Mapping):
        return {str(key): redact(item, parent_key=str(key)) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    return value


def _json_dump(value: Any, stream: Any = sys.stdout) -> None:
    json.dump(redact(value), stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")


def _is_loopback(hostname: str | None) -> bool:
    if not hostname:
        return False
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def normalize_base_url(value: str) -> str:
    candidate = value.strip().rstrip("/")
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise UsageFailure("--base-url must be an http(s) origin without embedded credentials")
    if parsed.query or parsed.fragment:
        raise UsageFailure("--base-url must not contain a query string or fragment")
    if parsed.path not in {"", "/"}:
        raise UsageFailure("--base-url must be an origin without a path")
    return candidate


def normalize_api_path(path: str) -> str:
    value = path.strip()
    if "#" in value:
        raise UsageFailure("API paths must not contain fragments")
    route, separator, query = value.partition("?")
    if re.search(r"%[0-9a-fA-F]{2}", route):
        raise UsageFailure("percent-encoded API route paths are not accepted")
    if "\\" in route or any(segment in {".", ".."} for segment in route.split("/")):
        raise UsageFailure("API route paths must not contain backslashes or dot segments")
    if not route.startswith("/"):
        route = "/" + route
    normalized = route if route.startswith("/api/") else "/api/v1" + route
    return normalized + (separator + query if separator else "")


def is_dangerous(method: str, path: str) -> bool:
    normalized = urlsplit(path).path.rstrip("/")
    action = normalized.rsplit("/", 1)[-1]
    return method.upper() == "DELETE" or action in DANGEROUS_ACTIONS


def parse_json_source(source: str | None, *, stdin: Any = sys.stdin) -> Any:
    if source is None:
        return None
    if source == "-":
        text = stdin.read()
    else:
        try:
            text = Path(source).read_text(encoding="utf-8")
        except OSError as exc:
            raise UsageFailure(f"cannot read JSON body from {source}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise UsageFailure(f"invalid JSON body: {exc}") from exc


def validate_output_destination(
    path: str,
    *,
    mode: int | None = None,
    require_new: bool = False,
) -> Path:
    destination = Path(path).expanduser()
    parent = destination.resolve(strict=False).parent
    if not parent.is_dir():
        raise UsageFailure(f"output directory does not exist: {parent}")
    if destination.is_symlink():
        raise UsageFailure(f"refusing to replace symlink output: {destination}")
    if destination.exists() and not destination.is_file():
        raise UsageFailure(f"output destination is not a regular file: {destination}")
    if require_new and destination.exists():
        raise UsageFailure(f"output already exists; choose a new path: {destination}")
    temporary_path: Path | None = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".probe", dir=parent)
        temporary_path = Path(temporary)
        try:
            if mode is not None:
                os.fchmod(fd, mode)
        finally:
            os.close(fd)
    except OSError as exc:
        raise UsageFailure(f"output directory is not writable for {destination}: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _atomic_write_chunks(
    path: Path,
    chunks: Iterable[bytes],
    *,
    mode: int | None = None,
    overwrite: bool = False,
) -> int:
    parent = path.expanduser().resolve(strict=False).parent
    if not parent.is_dir():
        raise UsageFailure(f"output directory does not exist: {parent}")
    if path.is_symlink():
        raise UsageFailure(f"refusing to replace symlink output: {path}")
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
    except OSError as exc:
        raise UsageFailure(f"cannot create temporary output for {path}: {exc}") from exc
    temporary_path = Path(temporary)
    size = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            if mode is not None:
                os.fchmod(handle.fileno(), mode)
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise UsageFailure("download stream returned a non-bytes chunk")
                handle.write(chunk)
                size += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, path)
        else:
            try:
                os.link(temporary_path, path)
            except FileExistsError as exc:
                raise UsageFailure(f"output already exists; choose a new path: {path}") from exc
            temporary_path.unlink()
        if mode is not None:
            os.chmod(path, mode)
        return size
    except OSError as exc:
        try:
            temporary_path.unlink(missing_ok=True)
        finally:
            raise UsageFailure(f"cannot write output {path}: {exc}") from exc
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        finally:
            raise


def _atomic_write(
    path: Path,
    data: bytes,
    *,
    mode: int | None = None,
    overwrite: bool = False,
) -> None:
    _atomic_write_chunks(path, (data,), mode=mode, overwrite=overwrite)


def write_secret_output(path: str, payload: Any) -> None:
    destination = validate_output_destination(
        path,
        mode=stat.S_IRUSR | stat.S_IWUSR,
        require_new=True,
    )
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(destination, encoded, mode=stat.S_IRUSR | stat.S_IWUSR)


def write_download(path: str, data: bytes, *, overwrite: bool = False) -> None:
    _atomic_write(Path(path).expanduser(), data, overwrite=overwrite)


def write_download_stream(path: str, chunks: Iterable[bytes], *, overwrite: bool = False) -> int:
    return _atomic_write_chunks(Path(path).expanduser(), chunks, overwrite=overwrite)


def response_payload(response: httpx.Response) -> Any:
    content_type = response.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            return response.json()
        except ValueError:
            return {"detail": "invalid_json_response", "body": response.text[:2048]}
    return response.text


def require_success(response: httpx.Response) -> httpx.Response:
    if response.is_success:
        return response
    payload = response_payload(response)
    detail = payload.get("detail", payload) if isinstance(payload, Mapping) else payload
    if response.status_code in {401, 403}:
        failure = AuthenticationFailure(f"HTTP {response.status_code}: {_detail_message(detail)}")
        failure.__cause__ = None
        raise failure
    raise ResponseFailure(response.status_code, detail)


def iter_sse(lines: Iterator[str]) -> Iterator[dict[str, str]]:
    event: dict[str, str] = {}
    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line:
            if event or data_lines:
                if data_lines:
                    event["data"] = "\n".join(data_lines)
                yield event
            event = {}
            data_lines = []
            continue
        if line.startswith(":"):
            yield {"comment": line[1:].lstrip()}
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            data_lines.append(value)
        elif field in {"event", "id", "retry"}:
            event[field] = value
    if event or data_lines:
        if data_lines:
            event["data"] = "\n".join(data_lines)
        yield event


class AlphaBrainAPIClient:
    def __init__(
        self,
        *,
        base_url: str,
        username: str | None = None,
        password_provider: Callable[[], str] | None = None,
        allow_insecure_http: bool = False,
        trust_env: bool = False,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if timeout <= 0:
            raise UsageFailure("request timeout must be positive")
        self.base_url = normalize_base_url(base_url)
        self.username = username.strip() if username else None
        self.password_provider = password_provider
        self.allow_insecure_http = allow_insecure_http
        self.timeout = timeout
        self.http = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            follow_redirects=False,
            trust_env=trust_env,
            transport=transport,
            headers={"Accept": "application/json", "User-Agent": "alphabrain-agent-api/1"},
        )
        self.deployment_mode: str | None = None
        self.csrf_token: str | None = None
        self.authenticated = False

    def __enter__(self) -> AlphaBrainAPIClient:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        self.http.close()

    def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self.http.request(method, path, **kwargs)
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            raise NetworkFailure(f"request failed: {exc}") from exc

    def setup_status(self) -> dict[str, Any]:
        response = require_success(self._send("GET", "/api/v1/setup/status"))
        payload = response.json()
        if not isinstance(payload, dict):
            raise ResponseFailure(response.status_code, "invalid_setup_status_response")
        self.deployment_mode = str(payload.get("deployment_mode", "personal"))
        return payload

    def status(self) -> dict[str, Any]:
        health = require_success(self._send("GET", "/api/v1/health")).json()
        setup = self.setup_status()
        return {"health": health, "setup": setup}

    def _check_sensitive_transport(self, purpose: str = "sensitive API requests") -> None:
        parsed = urlsplit(self.base_url)
        if parsed.scheme == "http" and not _is_loopback(parsed.hostname) and not self.allow_insecure_http:
            raise TransportSecurityFailure(
                f"refusing to send {purpose} over non-loopback HTTP; use HTTPS, an SSH tunnel, "
                "or explicitly pass --allow-insecure-http"
            )

    def _password(self) -> str:
        if self.password_provider is None:
            if not sys.stdin.isatty():
                raise AuthenticationFailure("laboratory mode requires --password-stdin or an interactive terminal")
            return getpass.getpass("AlphaBrain password: ")
        password = self.password_provider()
        if not password:
            raise AuthenticationFailure("empty password")
        return password

    def ensure_authenticated(self) -> None:
        if self.authenticated or self.deployment_mode == "personal":
            return
        status_payload = self.setup_status() if self.deployment_mode is None else None
        if status_payload is not None and not status_payload.get("initialized", False):
            raise AuthenticationFailure("AlphaBrain backend setup is required before protected API calls")
        if self.deployment_mode == "personal":
            self.authenticated = True
            return
        if self.deployment_mode != "lab":
            raise AuthenticationFailure(f"unsupported deployment mode: {self.deployment_mode}")
        self._check_sensitive_transport("laboratory credentials")
        username = self.username
        if not username:
            if not sys.stdin.isatty():
                raise AuthenticationFailure("laboratory mode requires --username")
            username = input("AlphaBrain username: ").strip()
        if not username:
            raise AuthenticationFailure("empty username")
        response = require_success(
            self._send(
                "POST",
                "/api/v1/auth/login",
                json={"username": username, "password": self._password()},
            )
        )
        payload = response.json()
        cookie_token = self.http.cookies.get("alphabrain_csrf")
        response_token = payload.get("csrf_token") if isinstance(payload, Mapping) else None
        self.csrf_token = str(cookie_token or response_token or "")
        if not self.csrf_token:
            raise AuthenticationFailure("login response did not provide a CSRF token")
        self.username = username
        self.authenticated = True

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        normalized_method, normalized_path, headers = self._prepare_request(
            method,
            path,
            headers=kwargs.pop("headers", None),
        )
        return require_success(self._send(normalized_method, normalized_path, headers=headers, **kwargs))

    def _prepare_request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[str, str, dict[str, str]]:
        normalized_method = method.upper()
        normalized_path = normalize_api_path(path)
        route_path = urlsplit(normalized_path).path
        if normalized_method in UNSAFE_METHODS:
            self._check_sensitive_transport()
        if route_path not in PUBLIC_API_PATHS:
            self.ensure_authenticated()
        prepared_headers = dict(headers or {})
        if (
            normalized_method in UNSAFE_METHODS
            and self.deployment_mode == "lab"
            and route_path not in CSRF_EXEMPT_PATHS
        ):
            if not self.csrf_token:
                self.ensure_authenticated()
            prepared_headers["X-CSRF-Token"] = str(self.csrf_token)
        return normalized_method, normalized_path, prepared_headers

    def download(
        self,
        method: str,
        path: str,
        *,
        output: str,
        json_body: Any = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        destination = validate_output_destination(output, require_new=not overwrite)
        normalized_method, normalized_path, headers = self._prepare_request(method, path)
        request_kwargs: dict[str, Any] = {"headers": headers}
        if json_body is not None:
            request_kwargs["json"] = json_body
        try:
            with self.http.stream(normalized_method, normalized_path, **request_kwargs) as response:
                if not response.is_success:
                    response.read()
                require_success(response)
                size = write_download_stream(str(destination), response.iter_bytes(), overwrite=overwrite)
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            raise NetworkFailure(f"download failed: {exc}") from exc
        return {"output": str(destination), "size_bytes": size}

    def request_value(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        secret_output: str | None = None,
    ) -> Any:
        if secret_output:
            self._check_sensitive_transport("secret responses")
        normalized_method, normalized_path, headers = self._prepare_request(method, path)
        request_kwargs: dict[str, Any] = {"headers": headers}
        if json_body is not None:
            request_kwargs["json"] = json_body
        try:
            with self.http.stream(normalized_method, normalized_path, **request_kwargs) as response:
                if not response.is_success:
                    response.read()
                    require_success(response)
                content_type = response.headers.get("content-type", "").lower()
                disposition = response.headers.get("content-disposition", "").lower()
                is_json = "application/json" in content_type
                if "attachment" in disposition or (not is_json and not content_type.startswith("text/")):
                    raise UsageFailure("file/binary responses require --output")
                response.read()
                payload = response_payload(response)
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            raise NetworkFailure(f"request failed: {exc}") from exc
        if secret_output:
            if not is_json:
                raise UsageFailure("--secret-output requires a JSON response")
            write_secret_output(secret_output, payload)
        return payload

    def openapi(self) -> dict[str, Any]:
        response = require_success(self._send("GET", "/api/openapi.json"))
        payload = response.json()
        if not isinstance(payload, dict):
            raise ResponseFailure(response.status_code, "invalid_openapi_response")
        return payload

    def schema(self, method: str | None = None, path: str | None = None) -> Any:
        document = self.openapi()
        if method is None and path is None:
            return document
        if not method or not path:
            raise UsageFailure("schema filtering requires both METHOD and PATH")
        normalized = normalize_api_path(path).split("?", 1)[0]
        operation = document.get("paths", {}).get(normalized, {}).get(method.lower())
        if operation is None:
            raise UsageFailure(f"OpenAPI operation not found: {method.upper()} {normalized}")
        return {"method": method.upper(), "path": normalized, "operation": operation}

    def _watch_success(self, *, kind: str, status: str, until: str | None) -> bool:
        target = until or WATCH_SPECS[kind].default_success_status
        if target == "terminal":
            return status in WATCH_SPECS[kind].terminal_statuses and status not in HARD_FAILURE_STATUSES
        return status == target

    def _check_watch_status(self, *, kind: str, item_id: str, status: str, until: str | None) -> dict[str, str] | None:
        if status == "deleted":
            raise WorkloadFailure(f"{kind} {item_id} was deleted while being watched")
        if status in HARD_FAILURE_STATUSES:
            raise WorkloadFailure(f"{kind} {item_id} reached unsuccessful status {status!r}")
        if self._watch_success(kind=kind, status=status, until=until):
            return {"id": item_id, "kind": kind, "status": status}
        if status in WATCH_SPECS[kind].terminal_statuses:
            target = until or WATCH_SPECS[kind].default_success_status
            raise WorkloadFailure(f"{kind} {item_id} reached {status!r} before target {target!r}")
        return None

    def _utility(self, item_id: str) -> dict[str, Any]:
        response = self.request("GET", "/utilities")
        payload = response.json()
        rows = payload if isinstance(payload, list) else payload.get("items", []) if isinstance(payload, Mapping) else []
        for row in rows:
            if isinstance(row, Mapping) and str(row.get("id")) == item_id:
                return dict(row)
        raise ResponseFailure(404, "utility_not_found")

    def watch(
        self,
        *,
        kind: str,
        item_id: str,
        until: str | None = None,
        timeout: float = 3600.0,
        follow_logs: bool = False,
        poll_interval: float = 1.0,
    ) -> dict[str, Any]:
        if kind not in WATCH_SPECS:
            raise UsageFailure(f"unknown workload kind: {kind}")
        if timeout <= 0:
            raise UsageFailure("watch timeout must be positive")
        self.ensure_authenticated()
        deadline = time.monotonic() + timeout
        spec = WATCH_SPECS[kind]
        if kind == "utility":
            while time.monotonic() < deadline:
                row = self._utility(item_id)
                status = str(row.get("status", ""))
                complete = self._check_watch_status(kind=kind, item_id=item_id, status=status, until=until)
                if complete:
                    return {**row, **complete}
                time.sleep(max(0.05, poll_interval))
            raise NetworkFailure(f"timed out waiting for utility {item_id}")

        assert spec.events_path is not None
        assert spec.detail_path is not None
        events_path = normalize_api_path(spec.events_path.format(id=item_id))
        last_event_id = ""
        consecutive_failures = 0
        while time.monotonic() < deadline:
            headers = {"Accept": "text/event-stream"}
            if last_event_id:
                headers["Last-Event-ID"] = last_event_id
            try:
                remaining = max(0.1, deadline - time.monotonic())
                stream_timeout = httpx.Timeout(
                    connect=min(self.timeout, remaining),
                    read=min(self.timeout, remaining),
                    write=min(self.timeout, remaining),
                    pool=min(self.timeout, remaining),
                )
                with self.http.stream("GET", events_path, headers=headers, timeout=stream_timeout) as response:
                    if not response.is_success:
                        response.read()
                    require_success(response)
                    received = False
                    for event in iter_sse(response.iter_lines()):
                        received = True
                        consecutive_failures = 0
                        if event.get("id"):
                            last_event_id = event["id"]
                        event_name = event.get("event", "message")
                        raw_data = event.get("data", "")
                        try:
                            data = json.loads(raw_data) if raw_data else {}
                        except json.JSONDecodeError:
                            data = {"text": raw_data}
                        if event_name == "log" and follow_logs:
                            text = data.get("text") or data.get("line") or raw_data
                            if text:
                                print(str(text), end="" if str(text).endswith("\n") else "\n", file=sys.stderr)
                        if event_name in {"metric", "progress"} and follow_logs:
                            print(json.dumps(redact(data), ensure_ascii=False), file=sys.stderr)
                        status = str(data.get("status", "")) if isinstance(data, Mapping) else ""
                        if status:
                            complete = self._check_watch_status(
                                kind=kind,
                                item_id=item_id,
                                status=status,
                                until=until,
                            )
                            if complete:
                                return complete
                        if time.monotonic() >= deadline:
                            break
                    if received:
                        consecutive_failures = 0
            except AuthenticationFailure:
                raise
            except ResponseFailure:
                raise
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    raise NetworkFailure(f"event stream repeatedly failed: {exc}") from exc

            if time.monotonic() >= deadline:
                break

            detail = self.request("GET", spec.detail_path.format(id=item_id)).json()
            status = str(detail.get("status", "")) if isinstance(detail, Mapping) else ""
            if status:
                complete = self._check_watch_status(kind=kind, item_id=item_id, status=status, until=until)
                if complete:
                    return {**dict(detail), **complete} if isinstance(detail, Mapping) else complete
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(max(0.1, poll_interval), remaining))
        raise NetworkFailure(f"timed out waiting for {kind} {item_id}")

    def infer(
        self,
        *,
        deployment_id: str,
        instructions: Sequence[str],
        image_paths: Sequence[str],
        states_path: str | None,
        save_inputs: bool,
        image_counts: Sequence[int] | None = None,
    ) -> Any:
        if not instructions:
            raise UsageFailure("infer requires at least one --instruction")
        data: dict[str, str] = {
            "instructions": json.dumps(list(instructions), ensure_ascii=False),
            "save_inputs": str(bool(save_inputs)).lower(),
        }
        if image_counts is not None:
            if len(image_counts) != len(instructions):
                raise UsageFailure("--image-count must be repeated once per instruction")
            if any(count < 0 for count in image_counts) or sum(image_counts) != len(image_paths):
                raise UsageFailure("--image-count values must be non-negative and sum to the number of images")
            data["image_counts"] = json.dumps(list(image_counts))
        if states_path:
            states = parse_json_source(states_path)
            data["states"] = json.dumps(states, ensure_ascii=False)
        handles: list[Any] = []
        files: list[tuple[str, tuple[str, Any, str]]] = []
        try:
            for raw_path in image_paths:
                path = Path(raw_path).expanduser()
                if not path.is_file():
                    raise UsageFailure(f"image does not exist: {path}")
                try:
                    handle = path.open("rb")
                except OSError as exc:
                    raise UsageFailure(f"cannot open image {path}: {exc}") from exc
                handles.append(handle)
                media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                files.append(("images", (path.name, handle, media_type)))
            response = self.request(
                "POST",
                f"/deployments/{deployment_id}/infer",
                data=data,
                files=files or None,
                timeout=max(self.timeout, 300.0),
            )
            return response.json()
        finally:
            for handle in handles:
                handle.close()


def _password_provider(password_stdin: bool, *, stdin: Any = sys.stdin) -> Callable[[], str] | None:
    if not password_stdin:
        return None
    consumed: list[str] = []

    def read_once() -> str:
        if not consumed:
            consumed.append(stdin.readline().rstrip("\r\n"))
        return consumed[0]

    return read_once


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("ALPHABRAIN_API_URL", DEFAULT_BASE_URL))
    parser.add_argument("--username", default=os.environ.get("ALPHABRAIN_API_USERNAME"))
    parser.add_argument("--password-stdin", action="store_true", help="Read the lab password from one stdin line")
    parser.add_argument(
        "--allow-insecure-http",
        action="store_true",
        help="Allow sensitive requests over non-loopback plain HTTP (unsafe; prefer HTTPS or an SSH tunnel)",
    )
    parser.add_argument(
        "--trust-env",
        action="store_true",
        help="Honor proxy and TLS environment variables (disabled by default)",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout in seconds")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("status", help="Show public backend and setup status")

    schema = commands.add_parser("schema", help="Read the live OpenAPI document or one operation")
    schema.add_argument("method", nargs="?")
    schema.add_argument("path", nargs="?")

    request = commands.add_parser("request", help="Send a generic API request")
    request.add_argument("method")
    request.add_argument("path")
    request.add_argument("--body", help="JSON file path, or - for stdin")
    request_outputs = request.add_mutually_exclusive_group()
    request_outputs.add_argument("--output", help="Atomically stream a response into a file")
    request_outputs.add_argument("--secret-output", help="Save complete JSON with mode 0600 to a new file")
    request.add_argument("--overwrite", action="store_true", help="Allow --output to replace an existing file")
    request.add_argument("--send", action="store_true", help="Required for every non-GET request")
    request.add_argument("--dangerous", action="store_true", help="Required for delete/control endpoints")

    watch = commands.add_parser("watch", help="Follow a managed workload until its target state")
    watch.add_argument("kind", choices=sorted(WATCH_SPECS))
    watch.add_argument("id")
    watch.add_argument("--until", help="Target status; defaults to completed, or running for deployments")
    watch.add_argument("--timeout", dest="watch_timeout", type=float, default=3600.0)
    watch.add_argument("--poll-interval", type=float, default=1.0)
    watch.add_argument("--follow-logs", action="store_true")

    infer = commands.add_parser("infer", help="Run managed multipart inference through the UI backend")
    infer.add_argument("deployment_id")
    infer.add_argument("--instruction", action="append", required=True)
    infer.add_argument("--image", action="append", default=[])
    infer.add_argument("--image-count", action="append", type=int, help="Images assigned to each instruction")
    infer.add_argument("--states", help="JSON file containing states")
    infer.add_argument("--save-inputs", action="store_true")
    return parser


def _render_response(response: httpx.Response, *, output: str | None, secret_output: str | None) -> Any:
    content_type = response.headers.get("content-type", "").lower()
    disposition = response.headers.get("content-disposition", "").lower()
    is_json = "application/json" in content_type
    if output:
        write_download(output, response.content)
        return {"output": str(Path(output).expanduser()), "size_bytes": len(response.content)}
    if "attachment" in disposition or (not is_json and not content_type.startswith("text/")):
        raise UsageFailure("file/binary responses require --output")
    payload = response_payload(response)
    if secret_output:
        if not is_json:
            raise UsageFailure("--secret-output requires a JSON response")
        write_secret_output(secret_output, payload)
    return payload


def run(args: argparse.Namespace) -> Any:
    if args.password_stdin and args.command == "request" and args.body == "-":
        raise UsageFailure("--password-stdin and --body - cannot share stdin")
    if args.password_stdin and args.command == "infer" and args.states == "-":
        raise UsageFailure("--password-stdin and --states - cannot share stdin")
    with AlphaBrainAPIClient(
        base_url=args.base_url,
        username=args.username,
        password_provider=_password_provider(args.password_stdin),
        allow_insecure_http=args.allow_insecure_http,
        trust_env=args.trust_env,
        timeout=args.timeout,
    ) as client:
        if args.command == "status":
            return client.status()
        if args.command == "schema":
            return client.schema(args.method, args.path)
        if args.command == "request":
            method = args.method.upper()
            normalized_path = normalize_api_path(args.path)
            if method != "GET" and not args.send:
                raise UsageFailure("non-GET requests require --send")
            if is_dangerous(method, normalized_path) and not args.dangerous:
                raise UsageFailure("delete/control requests require --dangerous")
            payload = parse_json_source(args.body)
            if args.overwrite and not args.output:
                raise UsageFailure("--overwrite requires --output")
            if args.output:
                return client.download(
                    method,
                    normalized_path,
                    output=args.output,
                    json_body=payload,
                    overwrite=args.overwrite,
                )
            if args.secret_output:
                validate_output_destination(
                    args.secret_output,
                    mode=stat.S_IRUSR | stat.S_IWUSR,
                    require_new=True,
                )
            return client.request_value(
                method,
                normalized_path,
                json_body=payload,
                secret_output=args.secret_output,
            )
        if args.command == "watch":
            return client.watch(
                kind=args.kind,
                item_id=args.id,
                until=args.until,
                timeout=args.watch_timeout,
                follow_logs=args.follow_logs,
                poll_interval=args.poll_interval,
            )
        if args.command == "infer":
            return client.infer(
                deployment_id=args.deployment_id,
                instructions=args.instruction,
                image_paths=args.image,
                states_path=args.states,
                save_inputs=args.save_inputs,
                image_counts=args.image_count,
            )
    raise UsageFailure(f"unknown command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
        _json_dump(result)
        return EXIT_OK
    except ClientFailure as exc:
        detail: dict[str, Any] = {"error": exc.__class__.__name__, "message": str(exc)}
        if isinstance(exc, ResponseFailure):
            detail.update({"status": exc.status_code, "detail": redact(exc.detail)})
        _json_dump(detail, stream=sys.stderr)
        return exc.exit_code
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        _json_dump({"error": exc.__class__.__name__, "message": str(exc)}, stream=sys.stderr)
        return EXIT_NETWORK
    except OSError as exc:
        _json_dump({"error": "LocalIOFailure", "message": str(exc)}, stream=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
