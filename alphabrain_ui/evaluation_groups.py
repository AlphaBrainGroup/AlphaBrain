"""Pure helpers for expanding grouped evaluation requests."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import EvaluationRequest


@dataclass(frozen=True)
class EvaluationChildSpec:
    position: int
    evaluation_kind: str
    name_suffix: str
    request: EvaluationRequest


def expand_evaluation_request(payload: EvaluationRequest) -> list[EvaluationChildSpec]:
    """Expand a public request into scheduler-atomic child runs."""

    if payload.kind == "standard":
        return [EvaluationChildSpec(0, "standard", "", payload)]
    if payload.kind == "batch":
        sources = (
            [None]
            if payload.source_kind == "managed_deployment"
            else payload.checkpoint_sources or [payload.checkpoint_source]
        )
        suites = payload.suites or [payload.suite or ""]
        children: list[EvaluationChildSpec] = []
        for source in sources:
            for suite in suites:
                child = payload.model_copy(
                    update={
                        "kind": "standard",
                        "checkpoint_source": source,
                        "checkpoint_sources": [],
                        "suite": suite or payload.suite,
                        "suites": [],
                        "resources": payload.resources.model_copy(
                            update={"gpu_count": 1, "gpu_ids": payload.resources.gpu_ids[:1]}
                        ),
                    }
                )
                if source is None:
                    suffix = suite or payload.benchmark_id
                else:
                    source_label = source.checkpoint_id if source.kind == "indexed" else source.path
                    suffix = f"{str(source_label).rstrip('/').split('/')[-1]} · {suite or payload.benchmark_id}"
                children.append(EvaluationChildSpec(len(children), "standard", suffix, child))
        return children
    if payload.kind == "online_stdp":
        child = payload.model_copy(update={"kind": "standard"})
        return [
            EvaluationChildSpec(0, "online_stdp_baseline", "Baseline", child),
            EvaluationChildSpec(1, "online_stdp_adapted", "Online STDP", child),
        ]
    return [EvaluationChildSpec(0, payload.kind, "", payload)]


__all__ = ["EvaluationChildSpec", "expand_evaluation_request"]
