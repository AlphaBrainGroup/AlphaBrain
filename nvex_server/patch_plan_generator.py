from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from .schemas import EvalRun, FailureCluster, PatchPlan, SourceRatio, TargetDataSpec


@dataclass(frozen=True)
class PatchRule:
    keyword: str
    training_strategy: str
    execution_backend: str
    annotation_schema: str
    verification_spec: str
    patch_episodes: int
    teleop_corrections: int
    lighting_variants: int = 0
    language_augmentations: int = 0
    source_ratio: tuple[float, float] = (0.7, 0.3)


RULES: tuple[PatchRule, ...] = (
    PatchRule(
        keyword="occlusion",
        training_strategy="continual_learning",
        execution_backend="alphabrain_cl",
        annotation_schema="occlusion_patch_v1",
        verification_spec="occlusion_robustness_eval",
        patch_episodes=120,
        teleop_corrections=20,
        lighting_variants=1,
    ),
    PatchRule(
        keyword="recovery",
        training_strategy="fine_tune",
        execution_backend="alphabrain_finetune",
        annotation_schema="recovery_fine_grained_v1",
        verification_spec="recovery_regression_eval",
        patch_episodes=80,
        teleop_corrections=40,
    ),
    PatchRule(
        keyword="language",
        training_strategy="vlm_cotrain",
        execution_backend="alphabrain_vlm_cotrain",
        annotation_schema="language_variation_v1",
        verification_spec="instruction_generalization_eval",
        patch_episodes=60,
        teleop_corrections=10,
        language_augmentations=120,
        source_ratio=(0.6, 0.4),
    ),
    PatchRule(
        keyword="lighting",
        training_strategy="continual_learning",
        execution_backend="alphabrain_cl",
        annotation_schema="lighting_shift_v1",
        verification_spec="appearance_shift_eval",
        patch_episodes=100,
        teleop_corrections=15,
        lighting_variants=3,
    ),
    PatchRule(
        keyword="long-horizon",
        training_strategy="world_model_verification",
        execution_backend="alphabrain_world_model",
        annotation_schema="long_horizon_debug_v1",
        verification_spec="rollout_verification_eval",
        patch_episodes=90,
        teleop_corrections=15,
    ),
    PatchRule(
        keyword="generalization",
        training_strategy="continual_learning",
        execution_backend="alphabrain_cl",
        annotation_schema="cross_robot_generalization_v1",
        verification_spec="cross_robot_generalization_eval",
        patch_episodes=140,
        teleop_corrections=20,
        source_ratio=(0.5, 0.5),
    ),
)


class PatchPlanGenerator:
    def generate(self, eval_run: EvalRun) -> PatchPlan:
        dominant_cluster = self._pick_dominant_cluster(eval_run)
        matched_rule = self._match_rule(dominant_cluster)
        expected_uplift = min(0.2, round(0.04 + dominant_cluster.share_of_failures * 0.18, 3))
        confidence = self._estimate_confidence(dominant_cluster)

        return PatchPlan(
            plan_id=f"plan_{uuid4().hex[:10]}",
            project_id=eval_run.project_id,
            based_on_eval_run=eval_run.run_id,
            root_causes=self._root_causes(eval_run),
            target_data_spec=TargetDataSpec(
                patch_episodes=matched_rule.patch_episodes,
                teleop_corrections=matched_rule.teleop_corrections,
                lighting_variants=matched_rule.lighting_variants,
                language_augmentations=matched_rule.language_augmentations,
            ),
            annotation_schema=matched_rule.annotation_schema,
            source_ratio=SourceRatio(real=matched_rule.source_ratio[0], synthetic=matched_rule.source_ratio[1]),
            training_strategy=matched_rule.training_strategy,
            execution_backend=matched_rule.execution_backend,
            verification_spec=matched_rule.verification_spec,
            expected_uplift=expected_uplift,
            confidence=confidence,
        )

    def _pick_dominant_cluster(self, eval_run: EvalRun) -> FailureCluster:
        if eval_run.failure_clusters:
            return max(eval_run.failure_clusters, key=lambda cluster: (cluster.share_of_failures, cluster.failure_count))

        return FailureCluster(
            cluster_id="cluster_fallback",
            label="General robustness gap",
            failure_pattern="occlusion",
            affected_tasks=[task.task_name for task in eval_run.task_breakdown if task.success_rate < eval_run.overall_success],
            share_of_failures=max(0.2, round(1.0 - eval_run.overall_success, 3)),
            failure_count=max(1, len(eval_run.task_breakdown)),
            severity="medium",
        )

    def _match_rule(self, cluster: FailureCluster) -> PatchRule:
        searchable_text = f"{cluster.label} {cluster.failure_pattern}".lower()
        for rule in RULES:
            if rule.keyword in searchable_text:
                return rule

        return RULES[0]

    def _root_causes(self, eval_run: EvalRun) -> list[str]:
        if eval_run.failure_clusters:
            return [cluster.failure_pattern for cluster in eval_run.failure_clusters]

        return ["under-specified failure patterns in imported EvalRun"]

    def _estimate_confidence(self, cluster: FailureCluster) -> float:
        severity_bonus = {
            "low": 0.05,
            "medium": 0.1,
            "high": 0.15,
            "critical": 0.2,
        }[cluster.severity]
        return min(0.95, round(0.55 + severity_bonus + cluster.share_of_failures * 0.2, 2))