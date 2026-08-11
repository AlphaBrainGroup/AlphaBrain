import asyncio
import hashlib
import json
import logging
from contextlib import asynccontextmanager

import pytest
import websockets.asyncio.client
import websockets.asyncio.server
import websockets.exceptions

from deployment.model_server.tools import msgpack_numpy
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


class FakePolicy:
    def __init__(self) -> None:
        self.calls = []

    def predict_action(self, **payload):
        self.calls.append(payload)
        return {"doubled": payload["value"] * 2}


@asynccontextmanager
async def _running_server(server: WebsocketPolicyServer):
    async with websockets.asyncio.server.serve(
        server._handler,
        "127.0.0.1",
        0,
        compression=None,
        max_size=None,
        process_request=server._process_request,
    ) as runtime:
        port = runtime.sockets[0].getsockname()[1]
        yield port


async def _get_health(port: int):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        f"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n".encode("ascii")
    )
    await writer.drain()
    raw = await reader.read()
    writer.close()
    await writer.wait_closed()

    head, body = raw.split(b"\r\n\r\n", 1)
    status_code = int(head.split(b" ", 2)[1])
    return status_code, json.loads(body), raw.decode("utf-8")


def test_authenticated_protocol_health_ping_and_infer(caplog):
    caplog.set_level(logging.INFO)

    async def scenario():
        plaintext_key = "fake-one-time-deployment-key"
        key_digest = hashlib.sha256(plaintext_key.encode("utf-8")).hexdigest()
        policy = FakePolicy()
        server = WebsocketPolicyServer(
            policy,
            metadata={"env": "test", "model": "fake"},
            api_key_sha256=key_digest,
            deployment_id="deployment-test-1",
        )

        async with _running_server(server) as port:
            status, initial_health, raw_health = await _get_health(port)
            assert status == 200
            assert initial_health["status"] == "ok"
            assert initial_health["deployment_id"] == "deployment-test-1"
            assert initial_health["metadata"] == {"env": "test", "model": "fake"}
            assert initial_health["uptime_seconds"] >= 0
            assert initial_health["last_inference_at"] is None
            assert initial_health["request_count"] == 0
            assert plaintext_key not in raw_health
            assert key_digest not in raw_health

            uri = f"ws://127.0.0.1:{port}"
            for headers in (None, {"Authorization": "Api-Key wrong-key"}):
                with pytest.raises(websockets.exceptions.InvalidStatus) as exc_info:
                    async with websockets.asyncio.client.connect(
                        uri,
                        additional_headers=headers,
                        proxy=None,
                    ):
                        pass
                assert exc_info.value.response.status_code == 401

            async with websockets.asyncio.client.connect(
                uri,
                additional_headers={"Authorization": f"Api-Key {plaintext_key}"},
                proxy=None,
            ) as websocket:
                metadata = msgpack_numpy.unpackb(await websocket.recv())
                assert metadata == {"env": "test", "model": "fake"}

                await websocket.send(msgpack_numpy.packb({"type": "ping", "request_id": "ping-1"}))
                ping = msgpack_numpy.unpackb(await websocket.recv())
                assert ping == {
                    "status": "ok",
                    "ok": True,
                    "type": "ping",
                    "request_id": "ping-1",
                }

                _, after_ping, _ = await _get_health(port)
                assert after_ping["last_inference_at"] is None
                assert after_ping["request_count"] == 0

                await websocket.send(
                    msgpack_numpy.packb(
                        {
                            "type": "infer",
                            "request_id": "infer-1",
                            "payload": {"value": 21},
                        }
                    )
                )
                inference = msgpack_numpy.unpackb(await websocket.recv())
                assert inference == {
                    "status": "ok",
                    "ok": True,
                    "type": "inference_result",
                    "request_id": "infer-1",
                    "data": {"doubled": 42},
                }

                _, after_inference, _ = await _get_health(port)
                assert after_inference["last_inference_at"].endswith("Z")
                assert after_inference["request_count"] == 1
                assert policy.calls == [{"value": 21}]

                await websocket.send(msgpack_numpy.packb({"type": "ping", "request_id": "ping-2"}))
                await websocket.recv()
                _, final_health, _ = await _get_health(port)
                assert final_health["last_inference_at"] == after_inference["last_inference_at"]
                assert final_health["request_count"] == 1

    asyncio.run(scenario())
    assert "fake-one-time-deployment-key" not in caplog.text
    assert hashlib.sha256(b"fake-one-time-deployment-key").hexdigest() not in caplog.text


def test_omitting_digest_preserves_legacy_unauthenticated_handshake():
    async def scenario():
        server = WebsocketPolicyServer(FakePolicy(), metadata={"legacy": True})
        async with _running_server(server) as port:
            async with websockets.asyncio.client.connect(f"ws://127.0.0.1:{port}", proxy=None) as websocket:
                assert msgpack_numpy.unpackb(await websocket.recv()) == {"legacy": True}

    asyncio.run(scenario())


def test_controller_digest_is_an_additional_non_public_credential():
    async def scenario():
        public_key = "public-deployment-key"
        controller_key = "ui-controller-key"
        server = WebsocketPolicyServer(
            FakePolicy(),
            metadata={"model": "fake"},
            api_key_sha256=hashlib.sha256(public_key.encode()).hexdigest(),
            controller_api_key_sha256=hashlib.sha256(controller_key.encode()).hexdigest(),
        )
        async with _running_server(server) as port:
            uri = f"ws://127.0.0.1:{port}"
            for key in (public_key, controller_key):
                async with websockets.asyncio.client.connect(
                    uri,
                    additional_headers={"Authorization": f"Api-Key {key}"},
                    proxy=None,
                ) as websocket:
                    assert msgpack_numpy.unpackb(await websocket.recv()) == {"model": "fake"}

            _, health, raw_health = await _get_health(port)
            assert health["status"] == "ok"
            assert controller_key not in raw_health
            assert hashlib.sha256(controller_key.encode()).hexdigest() not in raw_health

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid_digest", ["not-a-digest", "a" * 63, "g" * 64])
def test_invalid_configured_digest_fails_closed(invalid_digest):
    with pytest.raises(ValueError, match="64-character SHA-256"):
        WebsocketPolicyServer(FakePolicy(), api_key_sha256=invalid_digest)


def test_server_clis_add_deployment_options_without_changing_defaults():
    from deployment.model_server.server_policy import build_argparser as base_parser
    from deployment.model_server.server_policy_cosmos import build_argparser as cosmos_parser

    for parser in (base_parser(), cosmos_parser()):
        defaults = parser.parse_args([])
        assert defaults.host == "0.0.0.0"
        assert defaults.port == 10093
        assert defaults.api_key_sha256 is None
        assert defaults.controller_api_key_sha256 is None
        assert defaults.deployment_id is None

        configured = parser.parse_args(
            [
                "--host",
                "127.0.0.1",
                "--api-key-sha256",
                "a" * 64,
                "--controller-api-key-sha256",
                "b" * 64,
                "--deployment-id",
                "deployment-42",
            ]
        )
        assert configured.host == "127.0.0.1"
        assert configured.api_key_sha256 == "a" * 64
        assert configured.controller_api_key_sha256 == "b" * 64
        assert configured.deployment_id == "deployment-42"
