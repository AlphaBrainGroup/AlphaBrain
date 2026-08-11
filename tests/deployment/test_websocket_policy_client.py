from __future__ import annotations

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def test_websocket_client_uses_managed_policy_key_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("ALPHABRAIN_POLICY_API_KEY", "ab_ctl_environment")
    monkeypatch.setattr(
        WebsocketClientPolicy,
        "_wait_for_server",
        lambda _self: (object(), {}),
    )

    from_environment = WebsocketClientPolicy()
    explicit = WebsocketClientPolicy(api_key="ab_explicit")
    explicitly_disabled = WebsocketClientPolicy(api_key="")

    assert from_environment._api_key == "ab_ctl_environment"
    assert explicit._api_key == "ab_explicit"
    assert explicitly_disabled._api_key == ""
