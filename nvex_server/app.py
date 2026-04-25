from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from .patch_plan_generator import PatchPlanGenerator
from .schemas import (
    EvalRun,
    ImprovementReport,
    IterationArtifacts,
    IterationJob,
    IterationResultSummary,
    IterationStartRequest,
    PatchPlan,
    PlanGenerationRequest,
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
    poll_counts: dict[str, int] = field(default_factory=dict)


def create_app() -> FastAPI:
    app = FastAPI(title="Nvex Server", version="0.1.0")
    store = InMemoryStore()
    patch_plan_generator = PatchPlanGenerator()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/eval/import", response_model=EvalRun)
    def import_eval_run(eval_run: EvalRun) -> EvalRun:
        store.eval_runs[eval_run.run_id] = eval_run
        return eval_run

    @app.post("/api/plan/generate", response_model=PatchPlan)
    def generate_patch_plan(request: PlanGenerationRequest) -> PatchPlan:
        eval_run = request.eval_run
        if eval_run is None:
            eval_run = store.eval_runs.get(request.eval_run_id or "")

        if eval_run is None:
            raise HTTPException(status_code=404, detail="EvalRun not found")

        store.eval_runs[eval_run.run_id] = eval_run

        patch_plan = patch_plan_generator.generate(eval_run)
        store.patch_plans[patch_plan.plan_id] = patch_plan
        return patch_plan

    @app.post("/api/iteration/start", response_model=IterationJob)
    def start_iteration(request: IterationStartRequest) -> IterationJob:
        patch_plan = store.patch_plans.get(request.plan_id)
        if patch_plan is None:
            raise HTTPException(status_code=404, detail="PatchPlan not found")

        eval_run = store.eval_runs.get(patch_plan.based_on_eval_run)
        success_before = eval_run.overall_success if eval_run is not None else max(0.0, 1.0 - patch_plan.expected_uplift)
        success_after = min(1.0, round(success_before + patch_plan.expected_uplift, 3))
        iteration_id = f"iter_{uuid4().hex[:10]}"
        output_checkpoint = f"{request.checkpoint}_patched"

        job = IterationJob(
            iteration_id=iteration_id,
            project_id=patch_plan.project_id,
            plan_id=patch_plan.plan_id,
            based_on_checkpoint=request.checkpoint,
            status="queued",
            execution_backend=request.execution_backend or patch_plan.execution_backend,
            config=request.config,
            output_checkpoint=output_checkpoint,
            result_summary=IterationResultSummary(success_before=success_before, success_after=success_after),
            artifacts=IterationArtifacts(
                logs=[f"artifacts/{iteration_id}/train.log"],
                videos=[],
                eval_runs=[patch_plan.based_on_eval_run],
            ),
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        report = ImprovementReport(
            iteration_id=iteration_id,
            plan_id=patch_plan.plan_id,
            project_id=patch_plan.project_id,
            success_before=success_before,
            success_after=success_after,
            uplift=round(success_after - success_before, 3),
            summary=(
                f"Applied {patch_plan.training_strategy} via {job.execution_backend} against "
                f"{patch_plan.verification_spec}."
            ),
            assets_created=[
                ReusableAsset(
                    asset_id=f"asset_{uuid4().hex[:10]}",
                    type="recipe",
                    name=f"{patch_plan.annotation_schema}_{patch_plan.training_strategy}",
                    source_project=patch_plan.project_id,
                    reuse_count=0,
                    linked_iteration=iteration_id,
                    description="Auto-generated reusable recipe from the current patch plan.",
                )
            ],
        )

        store.iteration_jobs[iteration_id] = job
        store.reports[iteration_id] = report
        store.poll_counts[iteration_id] = 0
        return job

    @app.get("/api/iteration/{iteration_id}/status", response_model=IterationJob)
    def get_iteration_status(iteration_id: str) -> IterationJob:
        job = store.iteration_jobs.get(iteration_id)
        if job is None:
            raise HTTPException(status_code=404, detail="IterationJob not found")

        poll_count = store.poll_counts.get(iteration_id, 0) + 1
        store.poll_counts[iteration_id] = poll_count

        if job.status == "queued" and poll_count >= 1:
            job.status = "running"
            job.updated_at = utc_now()
        elif job.status == "running" and poll_count >= 2:
            job.status = "completed"
            job.updated_at = utc_now()

        return job

    @app.get("/api/report/{iteration_id}", response_model=ImprovementReport)
    def get_report(iteration_id: str) -> ImprovementReport:
        report = store.reports.get(iteration_id)
        if report is None:
            raise HTTPException(status_code=404, detail="ImprovementReport not found")
        return report

    return app


app = create_app()