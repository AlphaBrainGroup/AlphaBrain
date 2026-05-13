"""Common environment / logging bootstrap for RLT_a training & eval entries."""
import logging
import os
from pathlib import Path


def _load_env_from_repo_root() -> None:
    """Walk up from this file until a `.env` is found, then load it.

    Used so the entries work regardless of where they are launched from.
    """
    here = Path(__file__).resolve()
    for ancestor in [here.parent, *here.parents]:
        env_file = ancestor / ".env"
        if env_file.is_file():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    v_clean = v.strip().split("#")[0].strip()
                    os.environ.setdefault(k.strip(), v_clean)
            return


def setup() -> None:
    """Idempotent bootstrap: env vars + logging.

    Call once at the top of any RLT_a entry script (before importing torch/wandb).
    """
    _load_env_from_repo_root()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
