# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].


import argparse
import importlib.util
import logging
import socket

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    # Keep heavyweight model imports out of CLI parsing and ``--help``.
    import torch

    from AlphaBrain.model.framework.base_framework import BaseFramework

    attention_backend = args.attention_backend
    if attention_backend == "auto":
        attention_backend = None
        if importlib.util.find_spec("flash_attn") is None:
            attention_backend = "sdpa"
            logging.info("flash-attn is unavailable; using SDPA attention")

    vla = BaseFramework.from_pretrained(  # TODO should auto detect framework from model path
        args.ckpt_path,
        base_vlm_path=args.base_vlm_path,
        attention_backend=attention_backend,
    )

    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "simpler_env"},
        api_key_sha256=args.api_key_sha256,
        controller_api_key_sha256=args.controller_api_key_sha256,
        deployment_id=args.deployment_id,
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser(description="AlphaBrain WebSocket policy server")
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Address to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--base-vlm-path", default=None, help="Optional absolute local VLM override")
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "sdpa", "flash_attention_2", "eager"),
        default="auto",
    )
    parser.add_argument("--idle_timeout", type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument(
        "--api-key-sha256",
        default=None,
        help="Optional SHA-256 hex digest of the API key; omit to retain unauthenticated legacy mode",
    )
    parser.add_argument(
        "--controller-api-key-sha256",
        default=None,
        help="Optional SHA-256 digest for the non-user-facing UI controller credential",
    )
    parser.add_argument("--deployment-id", default=None, help="Deployment identifier reported by GET /healthz")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
