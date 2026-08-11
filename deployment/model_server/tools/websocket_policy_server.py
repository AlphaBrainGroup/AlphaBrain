# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""WebSocket policy server with optional deployment authentication and health checks."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import traceback
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Any

import websockets.asyncio.server
import websockets.frames

from . import msgpack_numpy

_LOGGER = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serve a policy over the AlphaBrain msgpack WebSocket protocol.

    Authentication is deliberately optional for backwards compatibility. When
    ``api_key_sha256`` is configured, clients must send this HTTP header during
    the WebSocket opening handshake::

        Authorization: Api-Key <plaintext-key>

    Only the SHA-256 digest is kept by this class. ``GET /healthz`` is always
    unauthenticated and contains deployment metadata only; it never includes
    key material.
    """

    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int | float = -1,
        metadata: dict | None = None,
        api_key_sha256: str | None = None,
        controller_api_key_sha256: str | None = None,
        deployment_id: str | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._deployment_id = deployment_id
        self._api_key_sha256 = self._normalize_api_key_sha256(api_key_sha256)
        self._controller_api_key_sha256 = self._normalize_api_key_sha256(
            controller_api_key_sha256
        )
        self._idle_timeout = idle_timeout

        now_monotonic = time.monotonic()
        self._started_at_monotonic = now_monotonic
        # The idle period begins when the service starts. Only inference
        # requests are allowed to move this timestamp forward.
        self._last_inference_monotonic = now_monotonic
        self._last_inference_at: float | None = None
        self._request_count = 0

        # WebSocket request headers (including Authorization) are only logged
        # by the library at DEBUG level. Keep the server logger at INFO even if
        # another package enables verbose logging globally.
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    @staticmethod
    def _normalize_api_key_sha256(value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        digest = value.strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("api_key_sha256 must be a 64-character SHA-256 hexadecimal digest")
        return digest

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._process_request,
        ) as server:
            if self._idle_timeout > 0:
                await self._idle_watchdog(server)
            else:
                await server.serve_forever()

    async def _idle_watchdog(self, server: websockets.asyncio.server.Server) -> None:
        """Close the service after a period without an inference request."""
        poll_interval = min(5.0, max(0.1, float(self._idle_timeout) / 4))
        while True:
            await asyncio.sleep(poll_interval)
            if time.monotonic() - self._last_inference_monotonic > self._idle_timeout:
                _LOGGER.info("Idle timeout (%ss) reached, shutting down server.", self._idle_timeout)
                server.close()
                await server.wait_closed()
                break

    async def _process_request(self, connection, request):
        """Serve health requests and authenticate WebSocket handshakes."""
        path = request.path.partition("?")[0]
        if path == "/healthz":
            payload = self._health_payload()
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
            response = connection.respond(HTTPStatus.OK, body)
            # ``respond`` creates a text/plain response. Replace, rather than
            # append, Content-Type to avoid duplicate headers.
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "application/json; charset=utf-8"
            response.headers["Cache-Control"] = "no-store"
            return response

        if self._api_key_sha256 is None:
            return None

        if not self._request_is_authorized(request.headers):
            response = connection.respond(HTTPStatus.UNAUTHORIZED, "Unauthorized\n")
            response.headers["WWW-Authenticate"] = "Api-Key"
            response.headers["Cache-Control"] = "no-store"
            return response

        return None

    def _request_is_authorized(self, headers) -> bool:
        try:
            values = headers.get_all("Authorization")
        except (AttributeError, KeyError):
            values = []
        if len(values) != 1:
            return False

        parts = values[0].strip().split(None, 1)
        if len(parts) != 2 or parts[0].lower() != "api-key" or not parts[1]:
            return False

        candidate_digest = hashlib.sha256(parts[1].encode("utf-8")).hexdigest()
        # compare_digest avoids leaking partial digest matches through timing.
        return any(
            hmac.compare_digest(candidate_digest, digest)
            for digest in (self._api_key_sha256, self._controller_api_key_sha256)
            if digest is not None
        )

    def _health_payload(self) -> dict[str, Any]:
        last_inference_at = None
        if self._last_inference_at is not None:
            last_inference_at = (
                datetime.fromtimestamp(self._last_inference_at, tz=timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )
        return {
            "status": "ok",
            "deployment_id": self._deployment_id,
            "metadata": self._metadata,
            "uptime_seconds": round(time.monotonic() - self._started_at_monotonic, 3),
            "last_inference_at": last_inference_at,
            "request_count": self._request_count,
        }

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection) -> None:
        _LOGGER.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()

        # Keep the existing first-frame metadata handshake intact.
        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                ret = self._route_message(msg)
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                _LOGGER.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _record_inference_request(self) -> None:
        now = time.time()
        self._last_inference_at = now
        self._last_inference_monotonic = time.monotonic()
        self._request_count += 1

    def _route_message(self, msg: dict) -> dict:
        """Route ping and inference requests without raising policy errors."""
        if not isinstance(msg, dict):
            return {
                "status": "error",
                "ok": False,
                "type": "inference_result",
                "request_id": "default",
                "error": {"message": "Payload must be a dict", "payload_type": str(type(msg))},
            }

        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")

        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        if mtype in {"infer", "predict_action"}:
            # Failed inference attempts still count as inference activity: the
            # deployment was actively used even if the policy rejected input.
            self._record_inference_request()
            try:
                # Explicit protocol envelopes use ``payload``. A flat dict is
                # retained as a backwards-compatible policy payload.
                if "payload" in msg:
                    payload = msg["payload"]
                else:
                    payload = {key: value for key, value in msg.items() if key not in {"type", "request_id"}}
                if not isinstance(payload, dict):
                    raise TypeError("Payload must be a dict")
                output_dict = self._policy.predict_action(**payload)
            except Exception as exc:
                _LOGGER.exception("Policy inference error (request_id=%s)", req_id)
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": str(exc)},
                }

            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "data": output_dict,
            }

        return {
            "status": "error",
            "ok": False,
            "type": "unknown",
            "request_id": req_id,
            "error": {"message": f"Unsupported message type '{mtype}'"},
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    raise NotImplementedError("This module is not intended to be run directly.")
