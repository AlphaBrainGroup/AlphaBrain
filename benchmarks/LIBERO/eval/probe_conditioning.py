#!/usr/bin/env python
"""Probe: does the policy server's output depend on the prompt?

Sends 4 inference requests against the running server:
    A1: prompt="open the middle drawer of the cabinet", image=A
    A2: prompt="put the bowl on the plate",            image=A   (same image, different prompt)
    B1: prompt="open the middle drawer of the cabinet", image=B   (different image, same prompt)
    B2: prompt="put the bowl on the plate",            image=B

If A1 ≈ A2 (same image, different prompt) → model ignores prompt
If A1 ≈ B1 (same prompt, different image) → model ignores image
If all four are essentially identical → model collapsed to a constant.

Run from repo root, vla env:
    python benchmarks/LIBERO/eval/probe_conditioning.py \
        --host 127.0.0.1 --port 5795
"""
import argparse
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def probe(client, prompt, img, state):
    payload = {
        "batch_images": [[img]],          # single view
        "instructions": [prompt],
        "states": [state.tolist()],
        "do_sample": False,
        "use_ddim": False,
        "num_ddim_steps": 10,
    }
    resp = client.infer(payload)
    if resp.get("status") == "error" or "data" not in resp:
        raise RuntimeError(f"server error: {resp}")
    a = np.asarray(resp["data"]["normalized_actions"], dtype=np.float32)
    if a.ndim == 3:
        a = a[0]  # (chunk, D)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5795)
    args = ap.parse_args()

    client = WebsocketClientPolicy(args.host, args.port)

    rng = np.random.RandomState(0)
    img_a = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img_b = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    # Use a plausible LIBERO start state (from cached demo)
    state = np.array([-0.203, 0.010, 1.178, 3.140, 0.004, -0.093, 0.0388, -0.0388],
                     dtype=np.float32)

    p1 = "open the middle drawer of the cabinet"
    p2 = "put the bowl on the plate"

    a1 = probe(client, p1, img_a, state)
    a2 = probe(client, p2, img_a, state)
    b1 = probe(client, p1, img_b, state)
    b2 = probe(client, p2, img_b, state)

    np.set_printoptions(precision=4, suppress=True)
    print(f"action shape: {a1.shape}")
    print(f"\n[A1] prompt={p1!r}, img=A\n  step0: {a1[0]}")
    print(f"[A2] prompt={p2!r}, img=A\n  step0: {a2[0]}")
    print(f"[B1] prompt={p1!r}, img=B\n  step0: {b1[0]}")
    print(f"[B2] prompt={p2!r}, img=B\n  step0: {b2[0]}")

    def diff(x, y, name):
        d = float(np.linalg.norm(x - y))
        per_dim = np.abs(x - y).max(axis=0)
        print(f"  {name}: chunk-wise L2={d:.5f}  max-per-dim-abs={per_dim}")

    print("\n=== diffs (chunk-wise) ===")
    diff(a1, a2, "A1 vs A2 (same image, diff prompt)")
    diff(a1, b1, "A1 vs B1 (diff image, same prompt)")
    diff(a1, b2, "A1 vs B2 (diff both)")
    diff(a2, b2, "A2 vs B2 (diff image, same prompt)")

    print("\n=== verdict ===")
    eps = 1e-3
    if float(np.linalg.norm(a1 - a2)) < eps:
        print("⚠ A1≈A2: model IGNORES PROMPT (same image+prompt → identical actions)")
    if float(np.linalg.norm(a1 - b1)) < eps:
        print("⚠ A1≈B1: model IGNORES IMAGE (same prompt+image → identical actions)")
    if float(np.linalg.norm(a1 - b2)) < eps:
        print("⚠ A1≈B2: model IGNORES BOTH (collapsed to constant)")


if __name__ == "__main__":
    main()
