import contextlib
import importlib.util
import io
import subprocess
import unittest
from pathlib import Path
from unittest import mock


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "parse_config.py"
_SPEC = importlib.util.spec_from_file_location("alphabrain_parse_config", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - importlib invariant
    raise RuntimeError(f"Unable to load {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
parse_config = _MODULE.parse_config


class EvalConfigContractTests(unittest.TestCase):
    def test_eval_shell_assignments_quote_untrusted_strings(self) -> None:
        checkpoint = "/tmp/checkpoint' ; echo CHECKPOINT_INJECTION; #"
        suite = "suite' ; echo SUITE_INJECTION; #"
        config = {
            "modes": {
                "fixture": {
                    "type": "eval",
                    "checkpoint": checkpoint,
                    "task_suite": suite,
                    "server_entrypoint": "deployment/model_server/server_policy.py",
                    "server_args": ["--label", "value' ; echo ARG_INJECTION; #"],
                    "task_ids": "0,4",
                    "task_limit": 1,
                    "seed": 9,
                    "num_views": 1,
                    "sort_tasks": False,
                }
            }
        }
        stdout = io.StringIO()
        with mock.patch("builtins.open", mock.mock_open(read_data="unused")):
            with mock.patch("yaml.safe_load", return_value=config):
                with contextlib.redirect_stdout(stdout):
                    parse_config("/ignored", "fixture")

        assignments = stdout.getvalue()
        command = (
            'eval "$1"; '
            'printf "%s\\n" "$EVAL_CHECKPOINT" "$TASK_SUITE" '
            '"$EVAL_TASK_IDS" "$EVAL_TASK_LIMIT" "$EVAL_SEED" "$EVAL_NUM_VIEWS" '
            '"$EVAL_SORT_TASKS"'
        )
        completed = subprocess.run(
            ["bash", "-c", command, "fixture", assignments],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.stdout.splitlines(),
            [checkpoint, suite, "0,4", "1", "9", "1", "false"],
        )

    def test_robocasa_vector_eval_seeds_workers_and_avoids_shared_recorders(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        for relative in (
            "benchmarks/Robocasa365/eval/simulation_env.py",
            "benchmarks/Robocasa_tabletop/eval/simulation_env.py",
        ):
            source = (repo_root / relative).read_text(encoding="utf-8")
            compile(source, relative, "exec")
            self.assertIn("self.env.reset(seed=config.seed)", source)
            self.assertIn("config.video.video_dir is not None and idx == 0", source)


if __name__ == "__main__":
    unittest.main()
