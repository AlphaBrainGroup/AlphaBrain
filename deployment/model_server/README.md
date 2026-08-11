# AlphaBrain model server

The model server exposes the existing msgpack policy protocol over WebSocket.
The first WebSocket frame remains the model metadata dictionary. Application
requests support `ping` and `infer` / `predict_action`.

## Start a policy server

The legacy command remains valid and starts an unauthenticated server on all
interfaces:

```bash
your_ckpt=./results/Checkpoints/1003_qwenfast/checkpoints/steps_50000_pytorch_model.pt

python deployment/model_server/server_policy.py \
    --ckpt_path "${your_ckpt}" \
    --port 10093 \
    --use_bf16
```

Managed deployments should set an explicit bind address, deployment ID, and
SHA-256 digest of a randomly generated API key. The server receives only the
digest; clients receive the plaintext key:

```bash
export ALPHABRAIN_DEPLOYMENT_API_KEY='replace-with-a-random-key'
api_key_sha256="$(printf '%s' "${ALPHABRAIN_DEPLOYMENT_API_KEY}" | sha256sum | awk '{print $1}')"

python deployment/model_server/server_policy.py \
    --ckpt_path "${your_ckpt}" \
    --host 127.0.0.1 \
    --port 10093 \
    --deployment-id deployment-example \
    --api-key-sha256 "${api_key_sha256}"
```

`server_policy_cosmos.py` accepts the same `--host`, `--port`,
`--deployment-id`, `--api-key-sha256`, and `--idle_timeout` options in addition
to its Cosmos checkpoint paths. Omitting `--api-key-sha256` intentionally keeps
the previous unauthenticated behavior.

## Health and clients

`GET /healthz` is available without authentication on the same port. It returns
only service state, deployment ID, public model metadata, uptime, last inference
time, and inference request count. Health checks and protocol `ping` requests do
not reset the inference idle timeout.

```bash
curl --fail http://127.0.0.1:10093/healthz

python deployment/model_server/tools/debug_server_policy.py \
    --host 127.0.0.1 \
    --port 10093 \
    --api-key "${ALPHABRAIN_DEPLOYMENT_API_KEY}" \
    --test init
```

Authenticated clients send the plaintext key only in the WebSocket opening
handshake:

```text
Authorization: Api-Key <plaintext-key>
```
