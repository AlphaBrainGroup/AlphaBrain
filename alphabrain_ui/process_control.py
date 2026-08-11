"""Cross-platform helpers for managed child processes."""

from __future__ import annotations

import os
import signal
import subprocess

import psutil

FORCE_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)


def new_process_group_kwargs() -> dict[str, object]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def process_group_matches(pid: int, pgid: int) -> bool:
    if os.name == "nt":
        return pid == pgid
    return os.getpgid(pid) == pgid


def signal_process_group(pid: int, pgid: int, sig: int) -> None:
    if os.name != "nt":
        os.killpg(pgid, sig)
        return
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess as exc:
        raise ProcessLookupError(pid) from exc
    processes = [*process.children(recursive=True), process]
    for item in reversed(processes):
        try:
            if sig == FORCE_KILL_SIGNAL:
                item.kill()
            else:
                item.terminate()
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise PermissionError(pid) from exc
