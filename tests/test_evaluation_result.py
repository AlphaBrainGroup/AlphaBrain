import json
import tempfile
import unittest
from pathlib import Path

from alphabrain_ui.evaluation_result import (
    EvaluationResultWriter,
    aggregate_results,
    mark_result_failed,
)


class EvaluationResultWriterTests(unittest.TestCase):
    def test_writer_updates_summary_and_uses_relative_video_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            result_path = root / "evaluation-result-v1.json"
            video_path = root / "videos" / "episode.mp4"
            video_path.parent.mkdir()
            video_path.write_bytes(b"fixture")

            writer = EvaluationResultWriter(
                result_path,
                benchmark="libero",
                checkpoint="checkpoint",
                suite={"name": "libero_goal"},
            )
            writer.record_episode(
                task_id="goal:0",
                task_name="open drawer",
                episode_index=0,
                success=True,
                steps=42,
                video_path=video_path,
            )
            writer.record_episode(
                task_id="goal:0",
                task_name="open drawer",
                episode_index=1,
                success=False,
            )
            writer.finish()

            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "evaluation-result-v1")
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["summary"]["num_tasks"], 1)
            self.assertEqual(payload["summary"]["num_episodes"], 2)
            self.assertEqual(payload["summary"]["num_successes"], 1)
            self.assertEqual(payload["summary"]["success_rate"], 0.5)
            self.assertEqual(payload["videos"][0]["path"], "videos/episode.mp4")

    def test_aggregate_rebases_artifacts_and_combines_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            sources = []
            for index, success in enumerate((True, False)):
                suite_dir = root / "suites" / str(index)
                result_path = suite_dir / "evaluation-result-v1.json"
                video_path = suite_dir / "videos" / f"{index}.mp4"
                video_path.parent.mkdir(parents=True)
                video_path.write_bytes(b"fixture")
                writer = EvaluationResultWriter(
                    result_path,
                    benchmark="libero",
                    checkpoint="checkpoint",
                    suite={"name": f"suite-{index}"},
                )
                writer.record_episode(
                    task_id=str(index),
                    task_name=f"task-{index}",
                    episode_index=0,
                    success=success,
                    video_path=video_path,
                )
                writer.finish()
                sources.append(result_path)

            output = root / "evaluation-result-v1.json"
            aggregate_results(output, sources, suite_name="libero_all")
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["num_tasks"], 2)
            self.assertEqual(payload["summary"]["num_episodes"], 2)
            self.assertEqual(payload["summary"]["success_rate"], 0.5)
            self.assertEqual(
                payload["videos"][0]["path"], "suites/0/videos/0.mp4"
            )

    def test_failure_preserves_partial_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            result_path = Path(temporary_dir) / "evaluation-result-v1.json"
            writer = EvaluationResultWriter(
                result_path,
                benchmark="robocasa365",
                checkpoint="checkpoint",
                suite={"name": "target50"},
            )
            writer.record_episode(
                task_id="task",
                task_name="task",
                episode_index=0,
                success=True,
            )
            mark_result_failed(
                result_path,
                benchmark="robocasa365",
                checkpoint="checkpoint",
                error="simulator crashed",
            )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["summary"]["num_episodes"], 1)
            self.assertEqual(payload["error"], "simulator crashed")


if __name__ == "__main__":
    unittest.main()
