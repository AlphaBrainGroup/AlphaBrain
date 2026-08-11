import ast
import importlib.util
import json
import math
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


# Load the stdlib-only helper directly so this focused test does not execute
# trainer_utils/__init__.py, whose existing logging setup requires Rich.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = (
    _REPO_ROOT
    / "AlphaBrain"
    / "training"
    / "trainer_utils"
    / "local_metrics.py"
)
_SPEC = importlib.util.spec_from_file_location("local_metrics_under_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
append_local_metrics = _MODULE.append_local_metrics


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class LocalMetricsTests(unittest.TestCase):
    def test_continual_and_all_rl_trainers_are_instrumented(self):
        expected_calls = {
            "AlphaBrain/training/train_alphabrain_vlm.py": 1,
            "AlphaBrain/training/continual_learning/train.py": 1,
            "AlphaBrain/training/reinforcement_learning/trainers/train_pretrain.py": 2,
            "AlphaBrain/training/reinforcement_learning/trainers/train_rlt_pretrain.py": 4,
            "AlphaBrain/training/reinforcement_learning/trainers/train_rl_onpolicy.py": 1,
            "AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py": 1,
            "AlphaBrain/training/reinforcement_learning/trainers/train_rl_grpo.py": 1,
            "AlphaBrain/training/reinforcement_learning/trainers/train_rl_vla_ppo.py": 1,
        }

        for relative_path, expected_count in expected_calls.items():
            source = (_REPO_ROOT / relative_path).read_text()
            tree = ast.parse(source, filename=relative_path)
            calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "append_local_metrics"
            ]
            self.assertEqual(
                len(calls),
                expected_count,
                f"unexpected local metrics coverage in {relative_path}",
            )
            for call in calls:
                keywords = {keyword.arg for keyword in call.keywords}
                self.assertIn("phase", keywords, relative_path)
                self.assertEqual(
                    len({"step", "iteration"} & keywords),
                    1,
                    f"metrics call needs one progress coordinate in {relative_path}",
                )

    def test_appends_timestamped_numeric_events(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics = {
                "loss": 1.25,
                "count": 3,
                "enabled": True,
                "tensor_scalar": _Scalar(2.5),
                "nested": {"task_0": 0.75},
                "not_numeric": "ignored",
                "not_scalar": [1, 2],
                "nan": math.nan,
                "inf": math.inf,
                "timestamp": 123,
                "phase": 123,
                "step": 123,
                "iteration": 123,
            }

            self.assertTrue(
                append_local_metrics(
                    tmp_dir,
                    metrics,
                    phase="continual_learning",
                    step=7,
                )
            )
            self.assertTrue(
                append_local_metrics(
                    tmp_dir,
                    {"reward": 1.0},
                    phase="rl_offpolicy",
                    iteration=8,
                )
            )

            lines = (Path(tmp_dir) / "metrics.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 2)

            first = json.loads(lines[0])
            self.assertEqual(first["phase"], "continual_learning")
            self.assertEqual(first["step"], 7)
            self.assertNotIn("iteration", first)
            self.assertEqual(first["metrics"]["loss"], 1.25)
            self.assertEqual(first["metrics"]["count"], 3)
            self.assertEqual(first["metrics"]["enabled"], 1)
            self.assertEqual(first["metrics"]["tensor_scalar"], 2.5)
            self.assertEqual(first["metrics"]["nested/task_0"], 0.75)
            self.assertNotIn("not_numeric", first["metrics"])
            self.assertNotIn("not_scalar", first["metrics"])
            self.assertNotIn("nan", first["metrics"])
            self.assertNotIn("inf", first["metrics"])
            datetime.fromisoformat(first["timestamp"].replace("Z", "+00:00"))

            second = json.loads(lines[1])
            self.assertEqual(second["phase"], "rl_offpolicy")
            self.assertEqual(second["iteration"], 8)
            self.assertEqual(second["metrics"]["reward"], 1.0)

            # The helper must never mutate dictionaries also sent to W&B.
            self.assertEqual(metrics["step"], 123)
            self.assertTrue(math.isnan(metrics["nan"]))

    def test_instrumentation_failures_do_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            not_a_directory = Path(tmp_dir) / "occupied"
            not_a_directory.write_text("file")
            self.assertFalse(
                append_local_metrics(
                    not_a_directory,
                    {"loss": 1.0},
                    phase="pretrain",
                    step=1,
                )
            )
            self.assertFalse(
                append_local_metrics(tmp_dir, {"loss": 1.0}, phase="", step=1)
            )
            self.assertFalse(
                append_local_metrics(tmp_dir, {"loss": 1.0}, phase="pretrain")
            )


if __name__ == "__main__":
    unittest.main()
