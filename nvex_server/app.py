from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .dispatcher import JobDispatcher
from .exporters import EvalArtifactExporter
from .patch_plan_generator import PatchPlanGenerator
from .schemas import (
    DemoStateResponse,
    EvalImportRequest,
    EvalRun,
    ImprovementReport,
    IterationJob,
    IterationStartRequest,
    PatchPlan,
    PlanGenerationRequest,
    PlatformMemorySnapshot,
    PlatformMemoryStats,
    ProjectContext,
    ReusableAsset,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class InMemoryStore:
    eval_runs: dict[str, EvalRun] = field(default_factory=dict)
    patch_plans: dict[str, PatchPlan] = field(default_factory=dict)
    iteration_jobs: dict[str, IterationJob] = field(default_factory=dict)
    reports: dict[str, ImprovementReport] = field(default_factory=dict)
    assets: dict[str, ReusableAsset] = field(default_factory=dict)
    demo_eval_run_id: str | None = None
    demo_patch_plan_id: str | None = None
    demo_iteration_id: str | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="Nvex Server", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    store = InMemoryStore()
    repo_root = Path(__file__).resolve().parents[1]
    exporter = EvalArtifactExporter()
    patch_plan_generator = PatchPlanGenerator()
    dispatcher = JobDispatcher(repo_root=repo_root, jobs_root=repo_root / "results" / "nvex_jobs")

    def build_report(
        iteration_id: str,
        patch_plan: PatchPlan,
        before_eval: EvalRun,
        after_eval: EvalRun | None,
    ) -> ImprovementReport:
        success_before = before_eval.overall_success
        success_after = after_eval.overall_success if after_eval is not None else min(
            1.0, round(success_before + patch_plan.expected_uplift, 3)
        )
        assets = [
            ReusableAsset(
                asset_id=f"asset_{uuid4().hex[:10]}",
                type="recipe",
                name=f"{patch_plan.annotation_schema}_{patch_plan.training_strategy}",
                source_project=patch_plan.project_id,
                reuse_count=0,
                linked_iteration=iteration_id,
                description="Auto-generated reusable recipe from the current patch plan.",
            )
        ]
        report = ImprovementReport(
            iteration_id=iteration_id,
            plan_id=patch_plan.plan_id,
            project_id=patch_plan.project_id,
            success_before=success_before,
            success_after=success_after,
            uplift=round(max(0.0, success_after - success_before), 3),
            summary=(
                f"Applied {patch_plan.training_strategy} via {patch_plan.execution_backend} and evaluated "
                f"against {patch_plan.verification_spec}."
            ),
            changes=[
                f"Targeted {patch_plan.target_data_spec.patch_episodes} patch episodes.",
                f"Added {patch_plan.target_data_spec.teleop_corrections} teleop correction trajectories.",
                f"Verification spec: {patch_plan.verification_spec}.",
            ],
            next_target="Temporal planning under language variation",
            assets_created=assets,
        )
        for asset in assets:
            store.assets[asset.asset_id] = asset
        return report

    def platform_memory() -> PlatformMemorySnapshot:
        recipe_names = sorted({asset.name for asset in store.assets.values() if asset.type == "recipe"})
        failure_names = sorted(
            {
                cluster.failure_pattern
                for eval_run in store.eval_runs.values()
                for cluster in eval_run.failure_clusters
            }
        )
        templates = sorted({patch_plan.annotation_schema for patch_plan in store.patch_plans.values()})
        return PlatformMemorySnapshot(
            recipes=recipe_names,
            templates=templates,
            failures=failure_names,
            stats=PlatformMemoryStats(
                recipes=len(recipe_names),
                templates=len(templates),
                patterns=len(failure_names),
                projects=len({eval_run.project_id for eval_run in store.eval_runs.values()}),
            ),
        )

    def project_context(eval_run: EvalRun, patch_plan: PatchPlan) -> ProjectContext:
        top_cluster = max(eval_run.failure_clusters, key=lambda item: item.share_of_failures) if eval_run.failure_clusters else None
        return ProjectContext(
            name="LIBERO Kitchen Pick-and-Place",
            checkpoint=eval_run.checkpoint or "ckpt_v0.7",
            domain="tabletop manipulation",
            suite=eval_run.benchmark_suite,
            status="Underperforming" if eval_run.overall_success < 0.7 else "Stable",
            status_note=top_cluster.label if top_cluster else "No dominant failure cluster",
            top_risk=top_cluster.failure_pattern if top_cluster else "generalization",
            next_action=f"Run {patch_plan.training_strategy} patch via {patch_plan.execution_backend}",
        )

    def save_eval_run(eval_run: EvalRun) -> EvalRun:
        store.eval_runs[eval_run.run_id] = eval_run
        return eval_run

    def import_eval_from_request(request: EvalImportRequest) -> EvalRun:
        if request.eval_run is not None:
            return save_eval_run(request.eval_run)

        eval_run = exporter.export(
            request.artifact_path or "",
            request.artifact_type,
            project_id=request.project_id,
            benchmark_suite=request.benchmark_suite,
            checkpoint=request.checkpoint,
            run_id=request.run_id,
        )
        return save_eval_run(eval_run)

    def generate_plan_from_request(request: PlanGenerationRequest) -> PatchPlan:
        eval_run = request.eval_run
        if eval_run is None:
            eval_run = store.eval_runs.get(request.eval_run_id or "")

        if eval_run is None:
            raise HTTPException(status_code=404, detail="EvalRun not found")

        save_eval_run(eval_run)
        patch_plan = patch_plan_generator.generate(eval_run)
        store.patch_plans[patch_plan.plan_id] = patch_plan
        return patch_plan

    def start_iteration_internal(request: IterationStartRequest) -> IterationJob:
        patch_plan = store.patch_plans.get(request.plan_id)
        if patch_plan is None:
            raise HTTPException(status_code=404, detail="PatchPlan not found")

        eval_run = store.eval_runs.get(patch_plan.based_on_eval_run)
        success_before = eval_run.overall_success if eval_run is not None else max(0.0, 1.0 - patch_plan.expected_uplift)
        job = dispatcher.start(patch_plan, request, before_success=success_before)

        after_path = request.config.get("after_eval_artifact_path")
        if after_path:
            after_eval_run = exporter.export(
                after_path,
                request.config.get("after_eval_artifact_type", "auto"),
                project_id=patch_plan.project_id,
                benchmark_suite=eval_run.benchmark_suite if eval_run is not None else None,
                checkpoint=job.output_checkpoint,
                run_id=f"{job.iteration_id}_after",
            )
            save_eval_run(after_eval_run)
            job.after_eval_run_id = after_eval_run.run_id
            job.result_summary.success_after = after_eval_run.overall_success
            job.artifacts.eval_runs.append(after_eval_run.run_id)
        report = build_report(job.iteration_id, patch_plan, eval_run or EvalRun(
            run_id=f"{job.iteration_id}_before",
            project_id=patch_plan.project_id,
            benchmark_suite="LIBERO_goal",
            overall_success=success_before,
        ), store.eval_runs.get(job.after_eval_run_id) if job.after_eval_run_id else None)

        store.iteration_jobs[job.iteration_id] = job
        store.reports[job.iteration_id] = report
        return job

    def seed_demo_state() -> None:
        before_path = repo_root / "nvex_server" / "examples" / "libero_kitchen_before_eval.json"
        after_path = repo_root / "nvex_server" / "examples" / "libero_kitchen_after_eval.json"
        before_eval = save_eval_run(
            exporter.export(
                str(before_path),
                "libero_eval_json",
                project_id="proj_libero_kitchen",
                benchmark_suite="LIBERO_goal",
                checkpoint="ckpt_v0.7",
                run_id="eval_libero_before",
            )
        )
        patch_plan = patch_plan_generator.generate(before_eval)
        store.patch_plans[patch_plan.plan_id] = patch_plan
        iteration = start_iteration_internal(
            IterationStartRequest(
                plan_id=patch_plan.plan_id,
                checkpoint=before_eval.checkpoint or "ckpt_v0.7",
                execution_backend=patch_plan.execution_backend,
                config={
                    "simulate": True,
                    "after_eval_artifact_path": str(after_path),
                    "after_eval_artifact_type": "libero_eval_json",
                },
            )
        )
        store.demo_eval_run_id = before_eval.run_id
        store.demo_patch_plan_id = patch_plan.plan_id
        store.demo_iteration_id = iteration.iteration_id

    seed_demo_state()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/eval/import", response_model=EvalRun)
    def import_eval_run(request: EvalImportRequest) -> EvalRun:
        return import_eval_from_request(request)

    @app.post("/api/plan/generate", response_model=PatchPlan)
    def generate_patch_plan(request: PlanGenerationRequest) -> PatchPlan:
        return generate_plan_from_request(request)

    @app.post("/api/iteration/start", response_model=IterationJob)
    def start_iteration(request: IterationStartRequest) -> IterationJob:
        return start_iteration_internal(request)

    @app.get("/api/iteration/{iteration_id}/status", response_model=IterationJob)
    def get_iteration_status(iteration_id: str) -> IterationJob:
        job = store.iteration_jobs.get(iteration_id)
        if job is None:
            raise HTTPException(status_code=404, detail="IterationJob not found")

        job = dispatcher.refresh(job)
        store.iteration_jobs[iteration_id] = job
        return job

    @app.get("/api/report/{iteration_id}", response_model=ImprovementReport)
    def get_report(iteration_id: str) -> ImprovementReport:
        report = store.reports.get(iteration_id)
        if report is None:
            raise HTTPException(status_code=404, detail="ImprovementReport not found")
        return report

    @app.get("/api/demo/state", response_model=DemoStateResponse)
    def get_demo_state() -> DemoStateResponse:
        if not store.demo_eval_run_id or not store.demo_patch_plan_id or not store.demo_iteration_id:
            raise HTTPException(status_code=500, detail="Demo state not initialized")

        eval_run = store.eval_runs[store.demo_eval_run_id]
        patch_plan = store.patch_plans[store.demo_patch_plan_id]
        iteration = store.iteration_jobs[store.demo_iteration_id]
        report = store.reports[store.demo_iteration_id]

        return DemoStateResponse(
            project=project_context(eval_run, patch_plan),
            current_eval_run=eval_run,
            patch_plan=patch_plan,
            iteration_job=iteration,
            report=report,
            platform_memory=platform_memory(),
        )

    return app


app = create_app()